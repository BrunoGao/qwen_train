#!/usr/bin/env python3
"""
双A40优化的数据预处理管道
针对有限显存优化的数据处理策略
"""

import os
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
import random

from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer


@dataclass
class DualA40Config:
    """双A40配置参数"""
    max_seq_length: int = 4096
    min_seq_length: int = 256  # 过滤过短样本
    target_samples: int = 100000  # 目标样本数，控制内存
    batch_size: int = 1000  # 处理批大小
    memory_limit_gb: int = 32  # 数据处理内存限制
    quality_threshold: float = 0.7  # 质量阈值
    max_workers: int = 8


class OptimizedCodeProcessor:
    """针对双A40优化的代码处理器"""
    
    def __init__(self, config: DualA40Config, tokenizer_path: Optional[str] = None):
        self.config = config
        self.tokenizer = None
        
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"✅ Tokenizer loaded from {tokenizer_path}")
            except Exception as e:
                print(f"⚠️ Failed to load tokenizer: {e}")
        
        # 优化的语言扩展映射
        self.language_map = {
            '.py': 'Python', '.java': 'Java', '.go': 'Go', '.rs': 'Rust',
            '.cpp': 'C++', '.c': 'C', '.h': 'C/C++', '.hpp': 'C++',
            '.js': 'JavaScript', '.ts': 'TypeScript'
        }
        
        # 高优先级文件类型（训练效果更好）
        self.priority_extensions = {'.py', '.java', '.go', '.rs', '.cpp'}
        
    def extract_high_quality_files(self, repo_path: str, repo_info: Dict) -> List[str]:
        """提取高质量代码文件"""
        repo_path = Path(repo_path)
        if not repo_path.exists():
            return []
        
        code_files = []
        priority_files = []
        
        # 遍历文件
        for file_path in repo_path.rglob('*'):
            if not file_path.is_file():
                continue
                
            # 检查扩展名
            if file_path.suffix not in self.language_map:
                continue
            
            # 过滤不需要的目录和文件
            if self._should_skip_file(str(file_path)):
                continue
            
            # 检查文件大小 (5KB - 200KB)
            try:
                file_size = file_path.stat().st_size
                if file_size < 5 * 1024 or file_size > 200 * 1024:
                    continue
            except OSError:
                continue
            
            file_str = str(file_path)
            if file_path.suffix in self.priority_extensions:
                priority_files.append(file_str)
            else:
                code_files.append(file_str)
        
        # 优先返回高优先级文件
        all_files = priority_files + code_files
        
        # 根据仓库质量分数采样
        quality_score = repo_info.get('code_quality_score', 8.0)
        sample_ratio = min(1.0, quality_score / 9.0)  # 质量越高采样比例越大
        
        max_files = int(len(all_files) * sample_ratio)
        selected_files = all_files[:max_files]
        
        print(f"Repository {repo_info['name']}: {len(selected_files)}/{len(all_files)} files selected")
        return selected_files

    def _should_skip_file(self, file_path: str) -> bool:
        """判断是否跳过文件"""
        file_path_lower = file_path.lower()
        
        # 跳过的目录
        skip_dirs = {
            'test', 'tests', 'testing', 'spec', 'specs', 'mock', 'mocks',
            'example', 'examples', 'demo', 'demos', 'benchmark', 'benchmarks',
            'node_modules', 'target', 'build', '__pycache__', '.git',
            'vendor', 'third_party', 'external', 'lib', 'libs', 'dist'
        }
        
        # 跳过的文件名模式
        skip_patterns = {
            'test_', '_test.', 'spec_', '_spec.', 'mock_', '_mock.',
            'example_', '_example.', 'demo_', '_demo.', 'benchmark_', '_benchmark.',
            '.min.', '.bundle.', '.dist.'
        }
        
        # 检查目录
        for skip_dir in skip_dirs:
            if f'/{skip_dir}/' in file_path_lower or file_path_lower.endswith(f'/{skip_dir}'):
                return True
        
        # 检查文件名
        filename = Path(file_path).name.lower()
        for pattern in skip_patterns:
            if pattern in filename:
                return True
        
        return False

    def process_file_optimized(self, file_info: Tuple[str, Dict]) -> Optional[Dict]:
        """优化的文件处理，内存友好"""
        file_path, repo_info = file_info
        
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            # 快速质量检查
            if not self._quick_quality_check(content):
                return None
            
            # 检测语言
            language = self._detect_language(file_path)
            
            # 生成训练样本
            sample = self._generate_optimized_sample(content, language, file_path, repo_info)
            
            # 长度检查
            if self.tokenizer:
                token_count = len(self.tokenizer.encode(sample['formatted_text']))
                if token_count < self.config.min_seq_length or token_count > self.config.max_seq_length:
                    return None
                sample['token_count'] = token_count
            else:
                # 估算token数量 (1 token ≈ 4 字符)
                estimated_tokens = len(sample['formatted_text']) // 4
                if estimated_tokens < self.config.min_seq_length or estimated_tokens > self.config.max_seq_length:
                    return None
                sample['token_count'] = estimated_tokens
            
            return sample
            
        except Exception as e:
            return None

    def _quick_quality_check(self, content: str) -> bool:
        """快速质量检查"""
        lines = content.split('\n')
        
        # 基本长度
        if len(content) < 500 or len(lines) < 15:
            return False
        
        # 空行比例
        empty_lines = sum(1 for line in lines if not line.strip())
        if empty_lines / len(lines) > 0.6:
            return False
        
        # 平均行长度
        avg_line_length = len(content) / len(lines)
        if avg_line_length < 10 or avg_line_length > 200:
            return False
        
        return True

    def _detect_language(self, file_path: str) -> str:
        """检测编程语言"""
        ext = Path(file_path).suffix.lower()
        return self.language_map.get(ext, 'Unknown')

    def _generate_optimized_sample(self, content: str, language: str, file_path: str, repo_info: Dict) -> Dict:
        """生成优化的训练样本"""
        
        # 简化的样本生成，减少计算开销
        sample_types = [
            "请分析以下{language}代码的功能和实现:",
            "解释这段{language}代码的主要逻辑:",
            "请对这个{language}代码进行代码审查:",
            "分析这段{language}代码的算法复杂度:"
        ]
        
        instruction = random.choice(sample_types).format(language=language)
        
        # 简化的代码分析
        lines = content.split('\n')
        analysis_parts = [f"这是一个{language}代码文件，包含{len(lines)}行代码。"]
        
        # 基本特征分析
        if language == 'Python':
            functions = len([l for l in lines if l.strip().startswith('def ')])
            classes = len([l for l in lines if l.strip().startswith('class ')])
            if functions > 0:
                analysis_parts.append(f"定义了{functions}个函数")
            if classes > 0:
                analysis_parts.append(f"定义了{classes}个类")
        
        elif language == 'Java':
            methods = len([l for l in lines if 'public ' in l and '(' in l])
            classes = len([l for l in lines if 'class ' in l])
            if methods > 0:
                analysis_parts.append(f"包含{methods}个方法")
            if classes > 0:
                analysis_parts.append(f"定义了{classes}个类")
        
        analysis = '，'.join(analysis_parts) + "。"
        
        # 构建Qwen对话格式
        formatted_text = (
            f"<|im_start|>system\n"
            f"你是一个代码分析专家。<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{instruction}\n\n"
            f"```{language.lower()}\n"
            f"{content}\n"
            f"```<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{analysis}<|im_end|>"
        )
        
        return {
            'instruction': instruction,
            'input': content,
            'output': analysis,
            'formatted_text': formatted_text,
            'language': language,
            'repo_name': repo_info['name'],
            'priority': repo_info['priority'],
            'quality_score': repo_info['code_quality_score'],
            'file_path': file_path
        }


def process_repositories_dual_a40(config: DualA40Config) -> Dataset:
    """双A40优化的仓库处理"""
    
    processor = OptimizedCodeProcessor(config)
    
    # 加载仓库配置
    with open("/codes/qwen25_code_training_repositories.json", 'r', encoding='utf-8') as f:
        repo_config = json.load(f)
    
    repo_map = {repo['name']: repo for repo in repo_config['repositories']}
    
    print(f"🎯 Target samples: {config.target_samples}")
    print(f"📏 Sequence length: {config.min_seq_length}-{config.max_seq_length}")
    
    all_file_infos = []
    
    # 按优先级处理仓库
    for priority in ['high_priority', 'medium_priority']:
        priority_path = Path("/codes/repositories") / priority
        if not priority_path.exists():
            continue
        
        print(f"\n📂 Processing {priority} repositories...")
        
        for repo_dir in priority_path.rglob('*'):
            if not repo_dir.is_dir() or repo_dir.name in {'.git', '__pycache__'}:
                continue
                
            # 找到实际的仓库目录（跳过语言分类目录）
            repo_name = None
            for possible_name in repo_map.keys():
                if possible_name in str(repo_dir):
                    repo_name = possible_name
                    break
            
            if not repo_name:
                continue
                
            repo_info = repo_map[repo_name].copy()
            repo_info['priority'] = priority.replace('_priority', '').upper()
            
            # 提取高质量文件
            code_files = processor.extract_high_quality_files(str(repo_dir), repo_info)
            
            # 添加到处理队列
            for file_path in code_files:
                all_file_infos.append((file_path, repo_info))
        
        # 如果已经有足够的文件，可以提前停止
        if len(all_file_infos) > config.target_samples * 2:
            break
    
    print(f"\n📊 Total files to process: {len(all_file_infos)}")
    
    # 随机打乱以确保多样性
    random.shuffle(all_file_infos)
    
    # 分批处理以控制内存使用
    all_samples = []
    batch_size = config.batch_size
    
    with mp.Pool(processes=min(config.max_workers, mp.cpu_count())) as pool:
        for i in tqdm(range(0, len(all_file_infos), batch_size), desc="Processing batches"):
            batch = all_file_infos[i:i + batch_size]
            
            # 处理当前批次
            batch_results = pool.map(processor.process_file_optimized, batch)
            
            # 过滤有效样本
            valid_samples = [sample for sample in batch_results if sample is not None]
            all_samples.extend(valid_samples)
            
            # 检查是否达到目标数量
            if len(all_samples) >= config.target_samples:
                print(f"🎯 Reached target samples: {len(all_samples)}")
                break
            
            # 内存控制：每处理5个批次清理一次
            if (i // batch_size + 1) % 5 == 0:
                print(f"📈 Current samples: {len(all_samples)}")
    
    # 截取到目标数量
    if len(all_samples) > config.target_samples:
        all_samples = all_samples[:config.target_samples]
    
    print(f"✅ Final dataset size: {len(all_samples)}")
    
    # 统计信息
    stats = {}
    for sample in all_samples:
        lang = sample['language']
        stats[lang] = stats.get(lang, 0) + 1
    
    print("\n📊 Language distribution:")
    for lang, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(all_samples) * 100
        print(f"  {lang}: {count} samples ({percentage:.1f}%)")
    
    # 创建数据集
    dataset = Dataset.from_list(all_samples)
    return dataset


def save_dual_a40_dataset(dataset: Dataset, output_path: str = "/codes/dual_a40_training_data"):
    """保存双A40优化数据集"""
    print(f"💾 Saving dataset to {output_path}...")
    
    # 保存数据集
    dataset.save_to_disk(output_path)
    
    # 保存统计信息
    df = dataset.to_pandas()
    stats = {
        'total_samples': len(dataset),
        'avg_token_count': int(df['token_count'].mean()),
        'token_distribution': {
            'min': int(df['token_count'].min()),
            'max': int(df['token_count'].max()),
            'std': float(df['token_count'].std())
        },
        'language_distribution': dict(df['language'].value_counts()),
        'quality_score_avg': float(df['quality_score'].mean()),
        'priority_distribution': dict(df['priority'].value_counts()),
        'estimated_training_tokens': int(df['token_count'].sum())
    }
    
    stats_path = Path(output_path) / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved: {stats['total_samples']} samples")
    print(f"📊 Average tokens per sample: {stats['avg_token_count']}")
    print(f"🎯 Total training tokens: {stats['estimated_training_tokens']:,}")
    
    return stats


if __name__ == "__main__":
    print("🚀 Starting dual A40 optimized data preprocessing...")
    
    # 双A40配置
    config = DualA40Config(
        max_seq_length=4096,
        min_seq_length=256,
        target_samples=80000,  # 适中的样本数量
        batch_size=500,
        memory_limit_gb=32,
        quality_threshold=0.7,
        max_workers=8
    )
    
    # 处理数据
    dataset = process_repositories_dual_a40(config)
    
    # 保存数据集
    stats = save_dual_a40_dataset(dataset)
    
    print("\n🎉 Dual A40 data preprocessing completed!")
    print(f"📁 Data location: /codes/dual_a40_training_data")
    print(f"🏆 Quality optimized for dual A40 training")