#!/bin/bash
# 双A40完整训练启动脚本

set -e

echo "🚀 Qwen 2.5 7B Code - Dual A40 Training Pipeline"
echo "=================================================="

# 配置参数
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/path/to/qwen2.5-7b-instruct}"
DATA_PATH="/codes/dual_a40_training_data"
OUTPUT_DIR="/codes/outputs/qwen_dual_a40_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/codes/logs"

# 创建必要目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0  # 启用P2P通信
export NCCL_IB_DISABLE=1   # 禁用InfiniBand
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 检查模型路径
if [ ! -d "$QWEN_MODEL_PATH" ]; then
    echo "❌ Model path not found: $QWEN_MODEL_PATH"
    echo "Please set QWEN_MODEL_PATH environment variable"
    exit 1
fi

echo "🔍 Configuration:"
echo "  Model: $QWEN_MODEL_PATH"
echo "  Data: $DATA_PATH"  
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# 步骤1: 数据预处理
echo "📊 Step 1: Data Preprocessing"
echo "=============================="
if [ ! -d "$DATA_PATH" ]; then
    echo "🔄 Preprocessing training data..."
    python /codes/dual_a40_data_processing.py 2>&1 | tee "$LOG_DIR/preprocessing.log"
    
    if [ $? -ne 0 ]; then
        echo "❌ Data preprocessing failed"
        exit 1
    fi
    echo "✅ Data preprocessing completed"
else
    echo "✅ Using existing preprocessed data: $DATA_PATH"
fi

# 检查数据
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Training data not found: $DATA_PATH"
    exit 1
fi

echo ""

# 步骤2: 环境检查
echo "🔍 Step 2: Environment Check" 
echo "============================"
python -c "
import torch
import transformers  
import deepspeed
import datasets

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ DeepSpeed: {deepspeed.__version__}')  
print(f'✅ Datasets: {datasets.__version__}')

if torch.cuda.is_available():
    print(f'✅ CUDA: {torch.version.cuda}')
    for i in range(min(torch.cuda.device_count(), 2)):
        props = torch.cuda.get_device_properties(i)
        print(f'✅ GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)')
else:
    print('❌ CUDA not available')
    exit(1)

try:
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('⚠️ Flash Attention not available')
"

if [ $? -ne 0 ]; then
    echo "❌ Environment check failed"
    exit 1
fi

echo ""

# 步骤3: 启动训练
echo "🎯 Step 3: Training"
echo "=================="
echo "⏰ Starting training at $(date)"

# 创建训练配置文件
cat > "$OUTPUT_DIR/training_config.json" << EOF
{
    "model_name_or_path": "$QWEN_MODEL_PATH",
    "data_path": "$DATA_PATH",
    "output_dir": "$OUTPUT_DIR",
    "num_train_epochs": 3.0,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1.5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "logging_steps": 50,
    "save_steps": 1000,
    "save_total_limit": 3,
    "bf16": true,
    "gradient_checkpointing": true,
    "dataloader_num_workers": 2,
    "remove_unused_columns": false,
    "report_to": "wandb",
    "run_name": "qwen-dual-a40-$(date +%Y%m%d-%H%M%S)",
    "deepspeed": "/codes/deepspeed_dual_a40.json",
    "max_seq_length": 4096,
    "use_flash_attention": true,
    "trust_remote_code": true
}
EOF

# 启动分布式训练
torchrun --nproc_per_node=2 \
         --master_port=29500 \
         /codes/train_dual_a40.py \
         --model_name_or_path "$QWEN_MODEL_PATH" \
         --data_path "$DATA_PATH" \
         --output_dir "$OUTPUT_DIR" \
         --num_train_epochs 3.0 \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 16 \
         --learning_rate 1.5e-5 \
         --weight_decay 0.01 \
         --warmup_steps 2000 \
         --logging_steps 50 \
         --save_steps 1000 \
         --save_total_limit 3 \
         --bf16 true \
         --gradient_checkpointing true \
         --dataloader_num_workers 2 \
         --remove_unused_columns false \
         --report_to wandb \
         --run_name "qwen-dual-a40-$(date +%Y%m%d-%H%M%S)" \
         --deepspeed /codes/deepspeed_dual_a40.json \
         --max_seq_length 4096 \
         --use_flash_attention true \
         --trust_remote_code true \
         2>&1 | tee "$LOG_DIR/training.log"

TRAINING_EXIT_CODE=$?

echo ""
echo "📋 Training Summary"
echo "=================="
echo "⏰ Training ended at $(date)"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    
    # 步骤4: 模型评估 (可选)
    echo ""
    echo "🧪 Step 4: Model Evaluation"
    echo "=========================="
    
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU Status after training:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    fi
    
    # 简单推理测试
    echo "🧪 Quick inference test..."
    python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    print('Loading model for testing...')
    tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_DIR', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        '$OUTPUT_DIR',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # 测试生成
    test_prompt = '请解释以下Python代码的功能：\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)'
    
    inputs = tokenizer(test_prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print('✅ Model inference test passed')
    print('📝 Test response:', response[:100] + '...' if len(response) > 100 else response)
    
except Exception as e:
    print(f'⚠️ Model test failed: {e}')
" 2>&1 | tee -a "$LOG_DIR/evaluation.log"
    
    echo ""
    echo "🎉 TRAINING PIPELINE COMPLETED!"
    echo "================================"
    echo "📁 Model: $OUTPUT_DIR"
    echo "📋 Logs: $LOG_DIR"
    echo "📊 Training log: $LOG_DIR/training.log"
    
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
    echo "📋 Check logs: $LOG_DIR/training.log"
    exit $TRAINING_EXIT_CODE
fi