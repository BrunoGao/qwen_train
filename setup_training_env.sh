#!/bin/bash
# Qwen 2.5 Code A40 训练环境配置脚本

set -e

echo "🚀 Setting up Qwen 2.5 Code training environment for A40 GPUs..."

# 检查CUDA版本
check_cuda() {
    echo "📋 Checking CUDA environment..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo "✅ NVIDIA drivers detected"
    else
        echo "❌ NVIDIA drivers not found. Please install CUDA drivers first."
        exit 1
    fi
    
    if command -v nvcc &> /dev/null; then
        nvcc --version
        echo "✅ CUDA toolkit detected"
    else
        echo "⚠️  CUDA toolkit not found. Some features may not work."
    fi
}

# 创建conda环境
setup_conda_env() {
    echo "🐍 Setting up Python environment..."
    
    # 检查conda是否存在
    if ! command -v conda &> /dev/null; then
        echo "❌ Conda not found. Please install Anaconda or Miniconda first."
        exit 1
    fi
    
    # 创建新环境
    ENV_NAME="qwen_code_training"
    echo "Creating conda environment: $ENV_NAME"
    
    conda create -n $ENV_NAME python=3.10 -y
    
    echo "Environment created. To activate: conda activate $ENV_NAME"
}

# 安装PyTorch和相关依赖
install_pytorch() {
    echo "🔥 Installing PyTorch and CUDA dependencies..."
    
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qwen_code_training
    
    # 安装PyTorch with CUDA 12.1
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    
    echo "✅ PyTorch installed successfully"
}

# 安装训练框架
install_training_frameworks() {
    echo "⚡ Installing training frameworks..."
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qwen_code_training
    
    # Core training dependencies
    pip install transformers==4.36.0
    pip install accelerate==0.25.0
    pip install deepspeed==0.12.6
    pip install peft==0.7.1
    
    # Flash Attention for A40
    pip install flash-attn==2.5.0 --no-build-isolation
    
    # Data processing
    pip install datasets==2.15.0
    pip install tokenizers==0.15.0
    
    # Monitoring and evaluation
    pip install wandb==0.16.0
    pip install tensorboard==2.15.0
    
    # Utilities
    pip install tqdm==4.66.0
    pip install pandas==2.1.0
    pip install numpy==1.24.0
    pip install scipy==1.11.0
    
    # Development tools
    pip install ipython==8.17.0
    pip install jupyter==1.0.0
    pip install matplotlib==3.8.0
    pip install seaborn==0.13.0
    
    echo "✅ Training frameworks installed successfully"
}

# 安装优化工具
install_optimization_tools() {
    echo "🛠️ Installing optimization tools..."
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qwen_code_training
    
    # Compilation optimization
    pip install ninja==1.11.0
    pip install packaging==23.2
    
    # Memory optimization
    pip install psutil==5.9.0
    pip install gpustat==1.1.1
    
    # Distributed training
    pip install mpi4py==3.1.5
    
    echo "✅ Optimization tools installed successfully"
}

# 配置环境变量
setup_environment_variables() {
    echo "🔧 Setting up environment variables..."
    
    # 创建环境变量文件
    cat > /codes/training_env.sh << 'EOF'
#!/bin/bash
# Qwen Code Training Environment Variables

# CUDA Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Distributed training
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export WORLD_SIZE=1
export RANK=0

# Paths
export QWEN_MODEL_PATH="/path/to/qwen2.5-7b-instruct"
export TRAINING_DATA_PATH="/codes/processed_training_data"
export OUTPUT_DIR="/codes/outputs"

# Optimization
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Wandb (optional)
export WANDB_PROJECT="qwen-code-training"
export WANDB_ENTITY="your-team"

echo "Environment variables set for Qwen Code training"
EOF
    
    chmod +x /codes/training_env.sh
    echo "✅ Environment variables configured in /codes/training_env.sh"
}

# 创建DeepSpeed配置
create_deepspeed_config() {
    echo "⚙️ Creating DeepSpeed configuration..."
    
    cat > /codes/deepspeed_config.json << 'EOF'
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-5,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": false,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "cpu_offload": false
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false,
  "dump_state": false
}
EOF
    
    echo "✅ DeepSpeed configuration created"
}

# 创建启动脚本
create_launch_scripts() {
    echo "🚀 Creating training launch scripts..."
    
    # 单机多卡训练脚本
    cat > /codes/launch_single_node.sh << 'EOF'
#!/bin/bash
# 单机多卡训练启动脚本

source /codes/training_env.sh

# 数据预处理
echo "Starting data preprocessing..."
python /codes/data_preprocessing.py

# 启动训练
echo "Starting Qwen 2.5 Code training on single node..."
torchrun --nproc_per_node=2 \
         --master_port=29500 \
         /codes/train_qwen_code.py \
         --model_name_or_path $QWEN_MODEL_PATH \
         --data_path $TRAINING_DATA_PATH \
         --output_dir $OUTPUT_DIR \
         --num_train_epochs 3 \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 8 \
         --learning_rate 2e-5 \
         --save_steps 1000 \
         --logging_steps 100 \
         --deepspeed /codes/deepspeed_config.json \
         --bf16 true \
         --gradient_checkpointing true \
         --dataloader_num_workers 4 \
         --report_to wandb \
         --run_name "qwen-2.5-7b-code-single-node"
EOF
    
    # 多机分布式训练脚本
    cat > /codes/launch_multi_node.sh << 'EOF'
#!/bin/bash
# 多机分布式训练启动脚本

source /codes/training_env.sh

# 获取节点信息
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-4}
MASTER_ADDR=${MASTER_ADDR:-"192.168.1.100"}

echo "Starting multi-node training..."
echo "Node rank: $NODE_RANK"
echo "Total nodes: $NNODES"
echo "Master address: $MASTER_ADDR"

# 启动训练
torchrun --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=29500 \
         --nproc_per_node=2 \
         /codes/train_qwen_code.py \
         --model_name_or_path $QWEN_MODEL_PATH \
         --data_path $TRAINING_DATA_PATH \
         --output_dir $OUTPUT_DIR \
         --num_train_epochs 3 \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 8 \
         --learning_rate 2e-5 \
         --save_steps 500 \
         --logging_steps 50 \
         --deepspeed /codes/deepspeed_config.json \
         --bf16 true \
         --gradient_checkpointing true \
         --dataloader_num_workers 4 \
         --report_to wandb \
         --run_name "qwen-2.5-7b-code-multi-node"
EOF
    
    chmod +x /codes/launch_*.sh
    echo "✅ Launch scripts created"
}

# 创建监控脚本
create_monitoring_scripts() {
    echo "📊 Creating monitoring scripts..."
    
    cat > /codes/monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""
训练监控脚本
"""
import time
import psutil
import subprocess
import json
from datetime import datetime

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for line in lines:
            parts = line.split(', ')
            gpu_info.append({
                'index': int(parts[0]),
                'name': parts[1],
                'memory_used': int(parts[2]),
                'memory_total': int(parts[3]),
                'utilization': int(parts[4]),
                'temperature': int(parts[5])
            })
        return gpu_info
    except:
        return []

def monitor_system():
    while True:
        timestamp = datetime.now().isoformat()
        
        # System info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU info
        gpu_info = get_gpu_info()
        
        # Create monitoring data
        monitor_data = {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'gpu_info': gpu_info
        }
        
        print(f"[{timestamp}] CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}% | GPUs: {len(gpu_info)}")
        for gpu in gpu_info:
            print(f"  GPU{gpu['index']}: {gpu['utilization']}% | {gpu['memory_used']}/{gpu['memory_total']}MB | {gpu['temperature']}°C")
        
        # Save to log file
        with open('/codes/logs/system_monitor.jsonl', 'a') as f:
            f.write(json.dumps(monitor_data) + '\n')
        
        time.sleep(30)  # Monitor every 30 seconds

if __name__ == "__main__":
    import os
    os.makedirs('/codes/logs', exist_ok=True)
    monitor_system()
EOF
    
    chmod +x /codes/monitor_training.py
    echo "✅ Monitoring script created"
}

# 创建测试脚本
create_test_scripts() {
    echo "🧪 Creating test scripts..."
    
    cat > /codes/test_environment.py << 'EOF'
#!/usr/bin/env python3
"""
环境测试脚本
"""
import torch
import transformers
import deepspeed
import datasets

def test_cuda():
    print("🔍 Testing CUDA...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

def test_torch():
    print("\n🔥 Testing PyTorch...")
    x = torch.randn(2, 3, device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn(3, 4, device='cuda' if torch.cuda.is_available() else 'cpu')
    z = torch.mm(x, y)
    print(f"Matrix multiplication test: {z.shape}")
    print("✅ PyTorch working correctly")

def test_transformers():
    print("\n🤖 Testing Transformers...")
    print(f"Transformers version: {transformers.__version__}")
    
    try:
        from transformers import AutoTokenizer
        print("✅ Transformers import successful")
    except Exception as e:
        print(f"❌ Transformers import failed: {e}")

def test_deepspeed():
    print("\n⚡ Testing DeepSpeed...")
    print(f"DeepSpeed version: {deepspeed.__version__}")
    
    try:
        # Simple DeepSpeed test
        print("✅ DeepSpeed import successful")
    except Exception as e:
        print(f"❌ DeepSpeed test failed: {e}")

def test_datasets():
    print("\n📊 Testing Datasets...")
    print(f"Datasets version: {datasets.__version__}")
    
    try:
        from datasets import Dataset
        test_data = Dataset.from_dict({"text": ["hello", "world"]})
        print(f"Test dataset created with {len(test_data)} examples")
        print("✅ Datasets working correctly")
    except Exception as e:
        print(f"❌ Datasets test failed: {e}")

def test_flash_attention():
    print("\n⚡ Testing Flash Attention...")
    try:
        import flash_attn
        print(f"Flash Attention version: {flash_attn.__version__}")
        print("✅ Flash Attention available")
    except ImportError:
        print("⚠️  Flash Attention not available (may affect training speed)")

if __name__ == "__main__":
    print("🧪 Testing Qwen Code training environment...\n")
    
    test_cuda()
    test_torch()
    test_transformers()
    test_deepspeed()
    test_datasets()
    test_flash_attention()
    
    print("\n🎉 Environment test completed!")
EOF
    
    chmod +x /codes/test_environment.py
    echo "✅ Test script created"
}

# 主安装流程
main() {
    echo "Starting Qwen 2.5 Code training environment setup..."
    
    check_cuda
    setup_conda_env
    install_pytorch
    install_training_frameworks
    install_optimization_tools
    setup_environment_variables
    create_deepspeed_config
    create_launch_scripts
    create_monitoring_scripts
    create_test_scripts
    
    echo ""
    echo "🎉 Environment setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment: conda activate qwen_code_training"
    echo "2. Source environment variables: source /codes/training_env.sh"
    echo "3. Test the environment: python /codes/test_environment.py"
    echo "4. Preprocess the data: python /codes/data_preprocessing.py"
    echo "5. Start training: bash /codes/launch_single_node.sh"
    echo ""
    echo "For multi-node training, run on each node:"
    echo "  NODE_RANK=<rank> NNODES=<total> MASTER_ADDR=<ip> bash /codes/launch_multi_node.sh"
    echo ""
    echo "Monitor training progress:"
    echo "  python /codes/monitor_training.py"
}

# 运行主函数
main