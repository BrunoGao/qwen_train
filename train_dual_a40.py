#!/usr/bin/env python3
"""
双A40专用Qwen 2.5 7B Code训练脚本
针对双GPU环境优化的训练实现
"""

import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    set_seed,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
import deepspeed
import wandb
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class DualA40ModelArguments:
    """双A40模型参数"""
    model_name_or_path: str = field(metadata={"help": "预训练模型路径"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Tokenizer名称"})
    use_flash_attention: bool = field(default=True, metadata={"help": "使用Flash Attention"})
    trust_remote_code: bool = field(default=True, metadata={"help": "信任远程代码"})


@dataclass 
class DualA40DataArguments:
    """双A40数据参数"""
    data_path: str = field(metadata={"help": "训练数据路径"})
    max_seq_length: int = field(default=4096, metadata={"help": "最大序列长度"})
    preprocessing_num_workers: int = field(default=2, metadata={"help": "数据预处理进程数"})


@dataclass
class DualA40TrainingArguments(TrainingArguments):
    """双A40训练参数"""
    output_dir: str = field(default="/codes/outputs/dual_a40")
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=1.5e-5)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=2000)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=2)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="wandb")
    run_name: str = field(default="qwen-dual-a40-code")
    deepspeed: str = field(default="/codes/deepspeed_dual_a40.json")
    
    # 双A40特定参数
    max_memory_per_gpu: int = field(default=43, metadata={"help": "每GPU最大内存GB"})
    memory_efficient: bool = field(default=True, metadata={"help": "内存高效模式"})
    checkpoint_activations: bool = field(default=True, metadata={"help": "检查点激活"})


class DualA40Dataset:
    """双A40优化的数据集类"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"📂 Loading dataset from {data_path}...")
        self.raw_dataset = load_from_disk(data_path)
        print(f"📊 Loaded {len(self.raw_dataset)} samples")
        
        # 设置特殊tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def get_tokenized_dataset(self):
        """获取tokenized数据集"""
        def tokenize_function(examples):
            # 使用预格式化的文本
            texts = examples['formatted_text']
            
            # 批量tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # 设置labels
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        print("🔄 Tokenizing dataset...")
        tokenized_dataset = self.raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=self.raw_dataset.column_names,
            desc="Tokenizing",
            load_from_cache_file=True
        )
        
        print(f"✅ Tokenization complete: {len(tokenized_dataset)} samples")
        return tokenized_dataset


class DualA40Trainer(Trainer):
    """双A40优化训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.step_times = []
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            # Shift tokens for causal LM
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        """训练步骤"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            self.accelerator.backward(loss)
        
        return loss.detach()
    
    def log(self, logs: Dict[str, float]) -> None:
        """增强的日志记录"""
        # 添加GPU内存使用信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logs[f'gpu_{i}_memory_allocated'] = round(memory_allocated, 2)
                logs[f'gpu_{i}_memory_reserved'] = round(memory_reserved, 2)
        
        # 添加训练速度信息
        if hasattr(self, 'state') and self.state.global_step > 0:
            elapsed_time = time.time() - self.start_time
            steps_per_second = self.state.global_step / elapsed_time
            logs['steps_per_second'] = round(steps_per_second, 4)
            
            # 估算剩余时间
            if self.state.max_steps > 0:
                remaining_steps = self.state.max_steps - self.state.global_step
                eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                logs['eta_hours'] = round(eta_seconds / 3600, 2)
        
        super().log(logs)


def setup_model_and_tokenizer(model_args: DualA40ModelArguments):
    """设置模型和tokenizer"""
    print("🤖 Loading model and tokenizer...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 模型配置
    model_config = {
        "torch_dtype": torch.bfloat16,
        "device_map": None,  # DeepSpeed处理设备映射
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    if model_args.use_flash_attention:
        model_config["attn_implementation"] = "flash_attention_2"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_config
    )
    
    # 调整vocabulary大小
    if len(tokenizer) != model.config.vocab_size:
        print(f"🔧 Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # 启用梯度检查点
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer


def setup_wandb(training_args):
    """设置Weights & Biases监控"""
    if training_args.report_to == "wandb":
        wandb.init(
            project="qwen-code-dual-a40",
            name=training_args.run_name,
            config={
                "model": "Qwen2.5-7B-Code",
                "hardware": "2x A40 48GB",
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "max_seq_length": 4096,
                "training_method": "Full Fine-tuning with DeepSpeed ZeRO-2"
            }
        )
        print("📊 Weights & Biases initialized")


def check_dual_a40_environment():
    """检查双A40环境"""
    print("🔍 Checking dual A40 environment...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available")
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"⚠️ Warning: Only {gpu_count} GPU(s) detected, expected 2")
    
    total_memory = 0
    for i in range(min(gpu_count, 2)):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        total_memory += memory_gb
        print(f"🎮 GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        if memory_gb < 40:
            print(f"⚠️ Warning: GPU {i} has less than 40GB memory")
    
    print(f"💾 Total GPU memory: {total_memory:.1f}GB")
    
    # 检查关键依赖
    try:
        import flash_attn
        print(f"⚡ Flash Attention: {flash_attn.__version__}")
    except ImportError:
        print("⚠️ Flash Attention not available")
    
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"🤗 Transformers: {transformers.__version__}")
    print(f"⚡ DeepSpeed: {deepspeed.__version__}")
    
    print("✅ Environment check completed")


def main():
    """主训练函数"""
    parser = HfArgumentParser((DualA40ModelArguments, DualA40DataArguments, DualA40TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    print("🚀 Starting Qwen 2.5 7B Code training on dual A40...")
    print(f"📁 Output directory: {training_args.output_dir}")
    
    # 环境检查
    check_dual_a40_environment()
    
    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # 准备数据集
    dataset_handler = DualA40Dataset(
        data_args.data_path,
        tokenizer, 
        max_length=data_args.max_seq_length
    )
    train_dataset = dataset_handler.get_tokenized_dataset()
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 设置监控
    setup_wandb(training_args)
    
    # 创建训练器
    trainer = DualA40Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 创建输出目录
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 开始训练
    print("🎯 Starting training...")
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time/3600:.2f} hours")
        
        # 保存模型
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # 保存训练指标
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["training_time_hours"] = training_time / 3600
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # 保存训练报告
        training_report = {
            "model_config": {
                "base_model": model_args.model_name_or_path,
                "hardware": "2x NVIDIA A40 48GB",
                "training_method": "Full Fine-tuning with DeepSpeed ZeRO-2"
            },
            "training_params": {
                "epochs": training_args.num_train_epochs,
                "batch_size_per_gpu": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * 2,
                "learning_rate": training_args.learning_rate,
                "max_seq_length": data_args.max_seq_length
            },
            "results": metrics,
            "dataset_info": {
                "total_samples": len(train_dataset),
                "data_path": data_args.data_path
            }
        }
        
        with open(Path(training_args.output_dir) / "training_report.json", "w") as f:
            json.dump(training_report, f, indent=2)
        
        print("🎉 Training completed successfully!")
        print(f"📊 Final loss: {metrics.get('train_loss', 'N/A')}")
        print(f"💾 Model saved to: {training_args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()