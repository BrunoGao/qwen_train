#!/bin/bash
# 双A40 GPU训练启动脚本

source /codes/training_env.sh

# 设置双卡环境
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0  # 启用P2P通信
export NCCL_IB_DISABLE=1

echo "🚀 Starting Qwen 2.5 Code training on 2x A40 GPUs..."
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# 数据预处理
if [ ! -d "/codes/processed_training_data" ]; then
    echo "📊 Preprocessing training data..."
    python /codes/data_preprocessing.py
fi

# 启动训练
torchrun --nproc_per_node=2 \
         --master_port=29500 \
         /codes/train_qwen_code.py \
         --model_name_or_path $QWEN_MODEL_PATH \
         --data_path /codes/processed_training_data \
         --output_dir /codes/outputs/qwen-2gpu \
         --num_train_epochs 3 \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 16 \
         --learning_rate 2e-5 \
         --save_steps 1000 \
         --logging_steps 50 \
         --deepspeed /codes/deepspeed_2gpu_config.json \
         --bf16 true \
         --gradient_checkpointing true \
         --dataloader_num_workers 2 \
         --model_max_length 4096 \
         --report_to wandb \
         --run_name "qwen-2.5-7b-code-2gpu"

echo "✅ Training completed!"