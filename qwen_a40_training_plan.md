# Qwen 2.5 7B Code A40 训练方案

## 数据分析总结
- **训练数据**: 2.8GB，155,271个代码文件
- **语言分布**: Python (6个项目), Java (5个), Go (5个), Rust (4个), C++ (5个)
- **代码质量**: 平均9.0分，最高9.8分(Rust语言项目)
- **项目特点**: 19个高优先级项目，6个中等优先级项目

## A40 GPU 训练架构设计

### 硬件资源规划
```yaml
GPU配置:
  - 型号: NVIDIA A40 48GB
  - 建议数量: 4-8块GPU (分布式训练)
  - 单机配置: 2x A40 (96GB总显存)
  - 多机配置: 4机x2卡 (384GB总显存)

CPU配置:
  - 核心数: 32-64核 per node
  - 内存: 256GB+ per node
  - 存储: NVMe SSD 2TB+ per node
```

### 模型配置优化
```python
# Qwen2.5 7B 针对A40优化的配置
model_config = {
    "model_name": "Qwen2.5-7B-Code",
    "vocab_size": 152064,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "max_position_embeddings": 131072,
    
    # A40优化参数
    "gradient_checkpointing": True,
    "use_cache": False,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2"
}
```

## 训练环境配置

### 1. 基础环境搭建
```bash
# CUDA 环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Python环境
conda create -n qwen_training python=3.10
conda activate qwen_training
```

### 2. 依赖安装
```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install deepspeed==0.12.6
pip install flash-attn==2.5.0
pip install datasets==2.15.0
pip install tokenizers==0.15.0
pip install wandb==0.16.0
pip install ninja packaging
```

### 3. 数据预处理管道

```python
# data_preprocessing.py
import os
import json
import multiprocessing as mp
from pathlib import Path
from tokenizers import Tokenizer
from datasets import Dataset

class CodeDataProcessor:
    def __init__(self, tokenizer_path, max_length=4096):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        
    def extract_code_files(self, repo_path):
        """提取代码文件并过滤"""
        extensions = {'.py', '.java', '.go', '.rs', '.cpp', '.c', '.h', '.hpp'}
        code_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # 过滤排除目录
            dirs[:] = [d for d in dirs if not any(
                exclude in root for exclude in 
                ['node_modules', 'target', 'build', '.git', '__pycache__']
            )]
            
            for file in files:
                if Path(file).suffix in extensions:
                    file_path = os.path.join(root, file)
                    if os.path.getsize(file_path) < 1024 * 1024:  # 限制1MB
                        code_files.append(file_path)
        
        return code_files
    
    def process_file(self, file_path):
        """处理单个代码文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 基础清理
            if len(content) < 100 or len(content) > 50000:
                return None
                
            # 代码质量过滤
            if content.count('\n') < 10:  # 至少10行
                return None
                
            # 构建训练样本
            language = self._detect_language(file_path)
            sample = {
                "instruction": f"请分析以下{language}代码的功能和实现:",
                "input": content,
                "output": self._generate_analysis(content, language),
                "language": language,
                "file_path": file_path
            }
            
            return sample
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def _detect_language(self, file_path):
        """检测编程语言"""
        ext_map = {
            '.py': 'Python', '.java': 'Java', '.go': 'Go',
            '.rs': 'Rust', '.cpp': 'C++', '.c': 'C',
            '.h': 'C', '.hpp': 'C++'
        }
        return ext_map.get(Path(file_path).suffix, 'Unknown')
    
    def _generate_analysis(self, content, language):
        """生成代码分析(简化版)"""
        lines = content.split('\n')
        functions = [l for l in lines if 'def ' in l or 'function ' in l or 'func ' in l]
        
        analysis = f"这是一个{language}代码文件，包含{len(lines)}行代码"
        if functions:
            analysis += f"，定义了{len(functions)}个函数"
        
        return analysis

# 数据预处理主流程
def preprocess_training_data():
    processor = CodeDataProcessor("/path/to/qwen_tokenizer")
    
    all_samples = []
    repo_base = "/codes/repositories"
    
    # 并行处理所有仓库
    for priority in ['high_priority', 'medium_priority']:
        priority_path = os.path.join(repo_base, priority)
        if os.path.exists(priority_path):
            for lang_dir in os.listdir(priority_path):
                lang_path = os.path.join(priority_path, lang_dir)
                if os.path.isdir(lang_path):
                    for repo_dir in os.listdir(lang_path):
                        repo_path = os.path.join(lang_path, repo_dir)
                        code_files = processor.extract_code_files(repo_path)
                        
                        # 多进程处理
                        with mp.Pool(mp.cpu_count()) as pool:
                            samples = pool.map(processor.process_file, code_files)
                            samples = [s for s in samples if s is not None]
                            all_samples.extend(samples)
    
    # 保存处理后的数据
    dataset = Dataset.from_list(all_samples)
    dataset.save_to_disk("/codes/processed_training_data")
    
    print(f"处理完成: {len(all_samples)} 个训练样本")
    return dataset
```

## 分布式训练策略

### 1. DeepSpeed配置
```json
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
    "enabled": false
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
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

### 2. 训练脚本
```python
# train_qwen_code.py
import os
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import deepspeed

class QwenCodeTrainer:
    def __init__(self, model_path, data_path, output_dir):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        
        # 初始化模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self):
        """准备训练数据"""
        dataset = load_from_disk(self.data_path)
        
        def tokenize_function(examples):
            # 构建对话格式
            conversations = []
            for i in range(len(examples['instruction'])):
                conversation = f"<|im_start|>system\n你是一个代码分析专家。<|im_end|>\n<|im_start|>user\n{examples['instruction'][i]}\n{examples['input'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
                conversations.append(conversation)
            
            # Tokenize
            tokenized = self.tokenizer(
                conversations,
                truncation=True,
                padding=False,
                max_length=4096,
                return_tensors=None
            )
            
            # 设置labels
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=8
        )
        
        return tokenized_dataset
    
    def train(self):
        """开始训练"""
        dataset = self.prepare_dataset()
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,
            fp16=False,
            bf16=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb",
            run_name="qwen-2.5-7b-code-training",
            deepspeed="/codes/deepspeed_config.json",
            gradient_checkpointing=True,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

# 启动训练
if __name__ == "__main__":
    trainer = QwenCodeTrainer(
        model_path="/path/to/qwen2.5-7b-instruct",
        data_path="/codes/processed_training_data",
        output_dir="/codes/outputs/qwen-code-finetuned"
    )
    trainer.train()
```

## 监控和评估

### 1. 训练监控
```bash
# 启动 wandb 监控
wandb login
export WANDB_PROJECT="qwen-code-training"
export WANDB_ENTITY="your-team"
```

### 2. 评估脚本
```python
# evaluation.py
def evaluate_code_model(model_path, test_data):
    """评估代码生成能力"""
    
    test_cases = [
        {
            "language": "Python",
            "task": "实现快速排序算法",
            "expected_keywords": ["def", "quicksort", "pivot", "recursive"]
        },
        {
            "language": "Java", 
            "task": "创建单例模式类",
            "expected_keywords": ["class", "private", "static", "getInstance"]
        },
        {
            "language": "Go",
            "task": "实现并发安全的计数器",
            "expected_keywords": ["func", "sync.Mutex", "goroutine"]
        }
    ]
    
    # 评估逻辑
    results = {}
    for case in test_cases:
        # 生成代码并评估
        pass
    
    return results
```

## 启动命令

### 单机多卡训练
```bash
torchrun --nproc_per_node=2 --master_port=29500 train_qwen_code.py
```

### 多机分布式训练
```bash
# 主节点
torchrun --nnodes=4 --node_rank=0 --master_addr="192.168.1.100" --master_port=29500 --nproc_per_node=2 train_qwen_code.py

# 其他节点
torchrun --nnodes=4 --node_rank=1 --master_addr="192.168.1.100" --master_port=29500 --nproc_per_node=2 train_qwen_code.py
```

## 预期结果
- **训练时间**: 单机2卡约48小时，4机8卡约12小时
- **显存使用**: 每卡约35-40GB (ZeRO Stage 2)
- **数据处理**: 约15万条高质量代码样本
- **模型性能**: 在代码理解、生成、补全任务上显著提升

## 注意事项
1. 确保A40驱动和CUDA版本兼容
2. 监控GPU温度和功耗
3. 定期保存checkpoints
4. 使用梯度累积平衡batch size和显存
5. 适当调整学习率和warmup步数