#!/usr/bin/env python3
"""
Qwen 2.5 Code A40 完整训练执行脚本
整合所有组件的主训练入口
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import transformers
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import deepspeed

# 导入自定义模块
from data_preprocessing import process_repositories, save_dataset
from train_qwen_code import main as train_main
from evaluation import run_comprehensive_evaluation
from checkpoint_manager import CheckpointManager


def setup_logging(log_dir: str = "/codes/logs"):
    """设置日志"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger


def check_environment():
    """检查训练环境"""
    logger = logging.getLogger(__name__)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPUs")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 检查关键依赖
    try:
        import flash_attn
        logger.info(f"Flash Attention: {flash_attn.__version__}")
    except ImportError:
        logger.warning("Flash Attention not available")
    
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Transformers: {transformers.__version__}")
    logger.info(f"DeepSpeed: {deepspeed.__version__}")
    
    return True


def validate_paths(args):
    """验证路径配置"""
    logger = logging.getLogger(__name__)
    
    # 检查模型路径
    if args.model_path and not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    # 检查数据路径
    if args.data_path:
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.warning(f"Data path not found: {data_path}, will create during preprocessing")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Path validation completed")


def run_data_preprocessing(args) -> str:
    """运行数据预处理"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")
    
    try:
        # 处理仓库数据
        dataset = process_repositories()
        
        # 保存数据集
        processed_data_path = args.data_path or "/codes/processed_training_data"
        stats = save_dataset(dataset, processed_data_path)
        
        logger.info(f"Data preprocessing completed: {stats['total_samples']} samples")
        return processed_data_path
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise


def run_training(args, data_path: str):
    """运行训练"""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # 设置环境变量
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': args.gpu_ids,
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'TOKENIZERS_PARALLELISM': 'false'
    })
    
    # 构建训练参数
    training_args = [
        "--model_name_or_path", args.model_path,
        "--data_path", data_path,
        "--output_dir", args.output_dir,
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--save_steps", str(args.save_steps),
        "--logging_steps", str(args.logging_steps),
        "--bf16", "true",
        "--gradient_checkpointing", "true",
        "--dataloader_num_workers", "4",
        "--remove_unused_columns", "false",
        "--deepspeed", "/codes/deepspeed_config.json"
    ]
    
    if args.use_wandb:
        training_args.extend([
            "--report_to", "wandb",
            "--run_name", f"qwen-code-{args.run_name or 'default'}"
        ])
    
    # 保存原始参数
    original_argv = sys.argv.copy()
    
    try:
        # 替换命令行参数
        sys.argv = ["train_qwen_code.py"] + training_args
        
        # 运行训练
        train_main()
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # 恢复原始参数
        sys.argv = original_argv


def run_evaluation(args):
    """运行模型评估"""
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    
    try:
        # 查找最新的模型
        model_path = args.output_dir
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        # 运行评估
        results = run_comprehensive_evaluation(
            model_path=model_path,
            output_dir=Path(args.output_dir) / "evaluation"
        )
        
        logger.info("Model evaluation completed")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def generate_training_report(args, training_time: float, evaluation_results: Optional[Dict] = None):
    """生成训练报告"""
    logger = logging.getLogger(__name__)
    
    report = {
        "training_info": {
            "model_name": "Qwen2.5-7B-Code",
            "base_model": args.model_path,
            "training_time_hours": round(training_time / 3600, 2),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gpu_config": args.gpu_ids,
            "output_dir": args.output_dir
        },
        "system_info": {
            "gpu_count": torch.cuda.device_count(),
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
    }
    
    if evaluation_results:
        report["evaluation_results"] = {
            "code_understanding": evaluation_results.get('code_understanding', {}).get('score', 0),
            "code_generation": evaluation_results.get('code_generation', {}).get('score', 0),
            "performance": evaluation_results.get('performance', {})
        }
    
    # 保存报告
    report_path = Path(args.output_dir) / "training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training report saved to: {report_path}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Training Time: {report['training_info']['training_time_hours']} hours")
    print(f"💾 Model Output: {args.output_dir}")
    if evaluation_results:
        print(f"🧠 Code Understanding: {report['evaluation_results']['code_understanding']:.2f}")
        print(f"⚡ Code Generation: {report['evaluation_results']['code_generation']:.2f}")
    print(f"📋 Full Report: {report_path}")
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen 2.5 Code A40 Training Pipeline")
    
    # 模型和数据配置
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--data_path", type=str, help="Processed data path (will create if not exists)")
    parser.add_argument("--output_dir", type=str, default="/codes/outputs", help="Output directory")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging every N steps")
    
    # 系统配置
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="GPU IDs to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # 运行配置
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--skip_training", action="store_true", help="Skip training")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--run_name", type=str, help="Run name for logging")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    # 设置随机种子
    set_seed(args.seed)
    
    try:
        # 环境检查
        logger.info("Checking environment...")
        check_environment()
        
        # 路径验证
        logger.info("Validating paths...")
        validate_paths(args)
        
        start_time = time.time()
        
        # 1. 数据预处理
        data_path = args.data_path
        if not args.skip_preprocessing:
            data_path = run_data_preprocessing(args)
        elif not data_path:
            data_path = "/codes/processed_training_data"
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Data path not found and preprocessing skipped: {data_path}")
        
        # 2. 模型训练
        if not args.skip_training:
            run_training(args, data_path)
        
        # 3. 模型评估
        evaluation_results = None
        if not args.skip_evaluation:
            evaluation_results = run_evaluation(args)
        
        # 4. 生成报告
        training_time = time.time() - start_time
        generate_training_report(args, training_time, evaluation_results)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import time
    main()