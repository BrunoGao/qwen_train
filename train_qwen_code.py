#!/usr/bin/env python3
"""
Qwen 2.5 7B Code A40 训练脚本
针对A40 GPU优化的分布式训练实现
"""

import os
import sys
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

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
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk, Dataset
import deepspeed


# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        metadata={"help": "预训练模型路径"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer路径，默认使用model_name_or_path"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "是否使用Flash Attention"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        metadata={"help": "训练数据路径"}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: int = field(
        default=8,
        metadata={"help": "数据预处理进程数"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """扩展的训练参数"""
    model_max_length: int = field(
        default=4096,
        metadata={"help": "模型最大长度"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用LoRA微调"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )


class QwenCodeDataset:
    """Qwen Code数据集处理"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据集
        print(f"Loading dataset from {data_path}...")
        self.dataset = load_from_disk(data_path)
        print(f"Loaded {len(self.dataset)} samples")
        
        # 设置特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def format_sample(self, sample: Dict[str, Any]) -> str:
        """格式化训练样本为对话格式"""
        instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']
        
        # 构建Qwen对话格式
        conversation = (
            f"<|im_start|>system\n"
            f"你是一个代码分析专家，专门帮助用户理解、分析和优化代码。\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{instruction}\n"
            f"```{sample.get('language', 'code')}\n"
            f"{input_text}\n"
            f"```\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{output_text}\n"
            f"<|im_end|>"
        )
        
        return conversation

    def tokenize_function(self, examples):
        """批量tokenize数据"""
        conversations = []
        
        for i in range(len(examples['instruction'])):
            sample = {
                'instruction': examples['instruction'][i],
                'input': examples['input'][i],
                'output': examples['output'][i],
                'language': examples.get('language', ['Unknown'])[i]
            }
            conversation = self.format_sample(sample)
            conversations.append(conversation)
        
        # Tokenize
        tokenized = self.tokenizer(
            conversations,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # 设置labels (对于因果语言模型，labels就是input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    def get_dataset(self):
        """获取处理后的数据集"""
        # 过滤过长的样本
        def filter_long_samples(example):
            text = self.format_sample(example)
            return len(self.tokenizer.encode(text)) <= self.max_length
        
        print("Filtering long samples...")
        filtered_dataset = self.dataset.filter(filter_long_samples)
        print(f"After filtering: {len(filtered_dataset)} samples")
        
        # Tokenize数据
        print("Tokenizing dataset...")
        tokenized_dataset = filtered_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=filtered_dataset.column_names,
            num_proc=8,
            desc="Tokenizing"
        )
        
        return tokenized_dataset


class QwenCodeTrainer(Trainer):
    """自定义Qwen Code训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # Shift so that tokens < n predict n
        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def setup_model_and_tokenizer(model_args: ModelArguments):
    """设置模型和tokenizer"""
    
    # 加载tokenizer
    tokenizer_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
        model_max_length=4096,
    )
    
    # 设置特殊tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 配置模型参数
    model_config = {
        "torch_dtype": torch.bfloat16,
        "device_map": None,  # 让DeepSpeed处理设备映射
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    if model_args.use_flash_attention:
        model_config["attn_implementation"] = "flash_attention_2"
    
    # 加载模型
    print(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_config
    )
    
    # 调整词汇表大小
    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    return model, tokenizer


def setup_lora(model, training_args):
    """设置LoRA微调"""
    if not training_args.use_lora:
        return model
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        print("LoRA configuration applied successfully")
        return model
        
    except ImportError:
        print("PEFT not available, skipping LoRA configuration")
        return model


def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # 记录训练参数
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 检测断点恢复
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # 设置LoRA
    model = setup_lora(model, training_args)
    
    # 准备数据集
    dataset_processor = QwenCodeDataset(
        data_args.data_path,
        tokenizer,
        max_length=data_args.max_seq_length
    )
    train_dataset = dataset_processor.get_dataset()
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 创建训练器
    trainer = QwenCodeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # 可以后续添加验证集
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # 保存模型
        
        # 保存训练指标
        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # 推送到Hub（可选）
    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()