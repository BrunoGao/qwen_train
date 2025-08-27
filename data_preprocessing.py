#!/usr/bin/env python3
"""
Qwen 2.5 Code 训练数据预处理管道
处理已下载的代码仓库，生成高质量的训练样本
"""

import os
import json
import re
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import hashlib

from tqdm import tqdm
import pandas as pd
from datasets import Dataset


@dataclass
class CodeSample:
    """代码样本数据结构"""
    instruction: str
    input: str
    output: str
    language: str
    file_path: str
    repo_name: str
    priority: str
    quality_score: float
    token_count: int


class CodeDataProcessor:
    """代码数据处理器"""
    
    def __init__(self, tokenizer_path: Optional[str] = None, max_length: int = 4096):
        self.max_length = max_length
        self.tokenizer = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                print(f"Warning: Failed to load tokenizer: {e}")
        
        # 代码文件扩展名映射
        self.language_extensions = {
            '.py': 'Python',
            '.java': 'Java', 
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.cc': 'C++',
            '.cxx': 'C++',
            '.c': 'C',
            '.h': 'C/C++',
            '.hpp': 'C++',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React TypeScript'
        }
        
        # 排除目录模式
        self.exclude_dirs = {
            'node_modules', 'target', 'build', '__pycache__', 
            '.git', '.svn', '.hg', 'vendor', 'third_party',
            'external', 'deps', 'lib', 'libs', 'dist'
        }
        
        # 排除文件模式
        self.exclude_files = {
            'test', 'spec', 'mock', 'fixture', 'example',
            'demo', 'benchmark', 'perf'
        }

    def extract_code_files(self, repo_path: str, repo_info: Dict) -> List[str]:
        """从仓库中提取代码文件"""
        code_files = []
        repo_path = Path(repo_path)
        
        if not repo_path.exists():
            print(f"Warning: Repository path not found: {repo_path}")
            return []
        
        print(f"Processing repository: {repo_info['name']}")
        
        for root, dirs, files in os.walk(repo_path):
            # 过滤排除目录
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d, root)]
            
            for file in files:
                file_path = Path(root) / file
                
                # 检查文件扩展名
                if file_path.suffix not in self.language_extensions:
                    continue
                    
                # 检查文件大小 (10KB - 1MB)
                try:
                    file_size = file_path.stat().st_size
                    if file_size < 10 * 1024 or file_size > 1024 * 1024:
                        continue
                except OSError:
                    continue
                
                # 过滤测试文件等
                if self._should_exclude_file(str(file_path)):
                    continue
                
                code_files.append(str(file_path))
        
        print(f"Found {len(code_files)} code files in {repo_info['name']}")
        return code_files

    def _should_exclude_dir(self, dirname: str, current_path: str) -> bool:
        """检查目录是否应该被排除"""
        dirname_lower = dirname.lower()
        return (
            dirname_lower in self.exclude_dirs or
            dirname_lower.startswith('.') or
            dirname_lower.endswith('test') or
            dirname_lower.endswith('tests') or
            'test' in dirname_lower
        )

    def _should_exclude_file(self, file_path: str) -> bool:
        """检查文件是否应该被排除"""
        file_path_lower = file_path.lower()
        filename = Path(file_path).stem.lower()
        
        return any(
            exclude in file_path_lower or exclude in filename
            for exclude in self.exclude_files
        )

    def process_file(self, file_info: Tuple[str, Dict]) -> Optional[CodeSample]:
        """处理单个代码文件"""
        file_path, repo_info = file_info
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            # 基本质量检查
            if not self._is_quality_code(content, file_path):
                return None
            
            # 检测编程语言
            language = self._detect_language(file_path)
            
            # 生成训练样本
            sample = self._generate_training_sample(
                content, language, file_path, repo_info
            )
            
            return sample
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def _is_quality_code(self, content: str, file_path: str) -> bool:
        """检查代码质量"""
        lines = content.split('\n')
        
        # 基本长度检查
        if len(content) < 200 or len(lines) < 10:
            return False
            
        # 检查空行比例
        empty_lines = sum(1 for line in lines if not line.strip())
        if empty_lines / len(lines) > 0.5:
            return False
            
        # 检查注释比例 (适量注释是好的)
        comment_lines = sum(1 for line in lines if self._is_comment_line(line, file_path))
        comment_ratio = comment_lines / len(lines)
        if comment_ratio > 0.8:  # 过多注释可能是文档文件
            return False
            
        # 检查代码复杂度指标
        if not self._check_code_complexity(content, file_path):
            return False
            
        return True

    def _is_comment_line(self, line: str, file_path: str) -> bool:
        """检查是否为注释行"""
        line = line.strip()
        if not line:
            return False
            
        ext = Path(file_path).suffix
        if ext == '.py':
            return line.startswith('#') or line.startswith('"""') or line.startswith("'''")
        elif ext in ['.java', '.cpp', '.c', '.h', '.hpp', '.js', '.ts']:
            return line.startswith('//') or line.startswith('/*')
        elif ext == '.go':
            return line.startswith('//') or line.startswith('/*')
        elif ext == '.rs':
            return line.startswith('//') or line.startswith('/*')
        
        return False

    def _check_code_complexity(self, content: str, file_path: str) -> bool:
        """检查代码复杂度"""
        ext = Path(file_path).suffix
        
        # 基本复杂度指标
        function_count = 0
        class_count = 0
        
        if ext == '.py':
            function_count = len(re.findall(r'\bdef\s+\w+', content))
            class_count = len(re.findall(r'\bclass\s+\w+', content))
        elif ext == '.java':
            function_count = len(re.findall(r'(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(', content))
            class_count = len(re.findall(r'\bclass\s+\w+', content))
        elif ext == '.go':
            function_count = len(re.findall(r'\bfunc\s+\w+', content))
            class_count = len(re.findall(r'\btype\s+\w+\s+struct', content))
        elif ext == '.rs':
            function_count = len(re.findall(r'\bfn\s+\w+', content))
            class_count = len(re.findall(r'\bstruct\s+\w+', content))
        elif ext in ['.cpp', '.c', '.h', '.hpp']:
            function_count = len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{', content))
            class_count = len(re.findall(r'\bclass\s+\w+', content))
        
        # 至少要有一些结构化内容
        return function_count > 0 or class_count > 0 or len(content.split('\n')) > 20

    def _detect_language(self, file_path: str) -> str:
        """检测编程语言"""
        ext = Path(file_path).suffix.lower()
        return self.language_extensions.get(ext, 'Unknown')

    def _generate_training_sample(self, content: str, language: str, file_path: str, repo_info: Dict) -> CodeSample:
        """生成训练样本"""
        
        # 生成多种类型的指令
        sample_types = [
            self._generate_code_analysis_sample,
            self._generate_code_explanation_sample,
            self._generate_code_review_sample,
            self._generate_function_extraction_sample
        ]
        
        # 随机选择样本类型
        import random
        sample_generator = random.choice(sample_types)
        instruction, output = sample_generator(content, language, file_path)
        
        # 计算token数量
        token_count = len(content.split()) if not self.tokenizer else len(
            self.tokenizer.encode(content, add_special_tokens=False)
        )
        
        return CodeSample(
            instruction=instruction,
            input=content,
            output=output,
            language=language,
            file_path=file_path,
            repo_name=repo_info['name'],
            priority=repo_info['priority'],
            quality_score=repo_info['code_quality_score'],
            token_count=token_count
        )

    def _generate_code_analysis_sample(self, content: str, language: str, file_path: str) -> Tuple[str, str]:
        """生成代码分析样本"""
        instruction = f"请分析以下{language}代码的功能、结构和关键特性："
        
        # 分析代码结构
        lines = content.split('\n')
        functions = self._extract_functions(content, language)
        classes = self._extract_classes(content, language)
        imports = self._extract_imports(content, language)
        
        output_parts = []
        output_parts.append(f"这是一个{language}代码文件，包含{len(lines)}行代码。")
        
        if imports:
            output_parts.append(f"导入了{len(imports)}个模块或库：{', '.join(imports[:5])}")
        
        if classes:
            output_parts.append(f"定义了{len(classes)}个类：{', '.join(classes[:3])}")
        
        if functions:
            output_parts.append(f"实现了{len(functions)}个函数：{', '.join(functions[:5])}")
        
        # 分析代码特点
        if 'async' in content or 'await' in content:
            output_parts.append("使用了异步编程模式。")
        
        if 'class' in content and ('def __init__' in content or 'constructor' in content):
            output_parts.append("包含面向对象编程特性。")
        
        output = ' '.join(output_parts)
        return instruction, output

    def _generate_code_explanation_sample(self, content: str, language: str, file_path: str) -> Tuple[str, str]:
        """生成代码解释样本"""
        instruction = f"请解释这段{language}代码的主要逻辑和实现原理："
        
        # 提取主要逻辑
        key_patterns = {
            'Python': [r'class\s+\w+', r'def\s+\w+', r'import\s+\w+', r'from\s+\w+'],
            'Java': [r'class\s+\w+', r'public\s+\w+', r'private\s+\w+', r'import\s+\w+'],
            'Go': [r'func\s+\w+', r'type\s+\w+', r'package\s+\w+', r'import\s+'],
            'Rust': [r'fn\s+\w+', r'struct\s+\w+', r'impl\s+\w+', r'use\s+\w+'],
            'C++': [r'class\s+\w+', r'int\s+\w+', r'void\s+\w+', r'#include']
        }
        
        patterns = key_patterns.get(language, [])
        key_elements = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            key_elements.extend(matches[:3])
        
        filename = Path(file_path).name
        output = f"这是一个名为{filename}的{language}源文件。"
        
        if key_elements:
            output += f"主要包含：{', '.join(key_elements)}。"
        
        output += f"代码结构清晰，遵循{language}的编程规范。"
        
        return instruction, output

    def _generate_code_review_sample(self, content: str, language: str, file_path: str) -> Tuple[str, str]:
        """生成代码审查样本"""
        instruction = f"请对这段{language}代码进行代码审查，指出优点和潜在改进点："
        
        output_parts = ["代码审查结果："]
        
        # 基本质量评估
        lines = content.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        if avg_line_length < 80:
            output_parts.append("✓ 代码行长度适中，易于阅读。")
        
        if re.search(r'(def|function|func|fn)\s+\w+', content):
            output_parts.append("✓ 代码具有良好的函数化结构。")
        
        # 检查注释
        comment_lines = sum(1 for line in lines if self._is_comment_line(line, file_path))
        if comment_lines > len(lines) * 0.1:
            output_parts.append("✓ 代码注释充分。")
        else:
            output_parts.append("• 建议添加更多注释以提高可读性。")
        
        # 检查命名规范
        if language == 'Python' and re.search(r'def\s+[a-z_][a-z0-9_]*', content):
            output_parts.append("✓ 遵循Python命名规范。")
        
        output = ' '.join(output_parts)
        return instruction, output

    def _generate_function_extraction_sample(self, content: str, language: str, file_path: str) -> Tuple[str, str]:
        """生成函数提取样本"""
        instruction = f"请从以下{language}代码中提取主要的函数和方法定义："
        
        functions = self._extract_functions(content, language)
        classes = self._extract_classes(content, language)
        
        output_parts = []
        if functions:
            output_parts.append(f"主要函数：{', '.join(functions[:10])}")
        if classes:
            output_parts.append(f"类定义：{', '.join(classes[:5])}")
        
        if not output_parts:
            output_parts.append(f"此{language}文件主要包含脚本代码，未发现明确的函数或类定义。")
        
        output = ' '.join(output_parts)
        return instruction, output

    def _extract_functions(self, content: str, language: str) -> List[str]:
        """提取函数名"""
        patterns = {
            'Python': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'Java': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            'Go': r'func\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'Rust': r'fn\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'C++': r'(?:\w+\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{',
            'C': r'(?:\w+\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{'
        }
        
        pattern = patterns.get(language, r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        matches = re.findall(pattern, content, re.MULTILINE)
        return [match if isinstance(match, str) else match[0] for match in matches]

    def _extract_classes(self, content: str, language: str) -> List[str]:
        """提取类名"""
        patterns = {
            'Python': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'Java': r'(?:public|private|protected)?\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'Go': r'type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+struct',
            'Rust': r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'C++': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        }
        
        pattern = patterns.get(language, r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        return re.findall(pattern, content)

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """提取导入语句"""
        patterns = {
            'Python': [r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)', r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import'],
            'Java': [r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*);'],
            'Go': [r'import\s+"([^"]*)"', r'import\s+([a-zA-Z_][a-zA-Z0-9_/]*)'],
            'Rust': [r'use\s+([a-zA-Z_][a-zA-Z0-9_:]*);'],
            'C++': [r'#include\s*[<"]([^>"]*)[>"]']
        }
        
        imports = []
        for pattern in patterns.get(language, []):
            imports.extend(re.findall(pattern, content))
        return imports


def process_repositories(base_path: str = "/codes/repositories") -> Dataset:
    """处理所有仓库生成训练数据"""
    
    processor = CodeDataProcessor()
    
    # 加载仓库信息
    with open("/codes/qwen25_code_training_repositories.json", 'r', encoding='utf-8') as f:
        repo_config = json.load(f)
    
    # 创建仓库映射
    repo_map = {repo['name']: repo for repo in repo_config['repositories']}
    
    all_file_infos = []
    total_files = 0
    
    # 遍历所有优先级和语言目录
    for priority in ['high_priority', 'medium_priority']:
        priority_path = Path(base_path) / priority
        if not priority_path.exists():
            continue
            
        for lang_dir in priority_path.iterdir():
            if not lang_dir.is_dir():
                continue
                
            for repo_dir in lang_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                    
                repo_name = repo_dir.name
                if repo_name not in repo_map:
                    print(f"Warning: Repository {repo_name} not found in config")
                    continue
                
                repo_info = repo_map[repo_name].copy()
                repo_info['priority'] = priority.replace('_priority', '').upper()
                
                # 提取代码文件
                code_files = processor.extract_code_files(str(repo_dir), repo_info)
                
                # 创建文件信息
                for file_path in code_files:
                    all_file_infos.append((file_path, repo_info))
                    
                total_files += len(code_files)
    
    print(f"Total files to process: {total_files}")
    
    # 并行处理所有文件
    print("Processing code files...")
    with mp.Pool(processes=min(mp.cpu_count(), 16)) as pool:
        results = list(tqdm(
            pool.imap(processor.process_file, all_file_infos),
            total=len(all_file_infos),
            desc="Processing files"
        ))
    
    # 过滤有效样本
    valid_samples = [sample for sample in results if sample is not None]
    print(f"Generated {len(valid_samples)} valid samples from {total_files} files")
    
    # 转换为字典格式
    data_dicts = []
    for sample in valid_samples:
        data_dicts.append({
            'instruction': sample.instruction,
            'input': sample.input,
            'output': sample.output,
            'language': sample.language,
            'file_path': sample.file_path,
            'repo_name': sample.repo_name,
            'priority': sample.priority,
            'quality_score': sample.quality_score,
            'token_count': sample.token_count
        })
    
    # 创建数据集
    dataset = Dataset.from_list(data_dicts)
    
    # 统计信息
    stats = defaultdict(int)
    for sample in valid_samples:
        stats[sample.language] += 1
    
    print("\nDataset Statistics:")
    for lang, count in sorted(stats.items()):
        print(f"  {lang}: {count} samples")
    
    return dataset


def save_dataset(dataset: Dataset, output_path: str = "/codes/processed_training_data"):
    """保存处理后的数据集"""
    print(f"Saving dataset to {output_path}...")
    dataset.save_to_disk(output_path)
    
    # 保存统计信息
    stats_path = Path(output_path) / "dataset_stats.json"
    stats = {
        'total_samples': len(dataset),
        'languages': dict(dataset.to_pandas()['language'].value_counts()),
        'priorities': dict(dataset.to_pandas()['priority'].value_counts()),
        'average_quality_score': float(dataset.to_pandas()['quality_score'].mean()),
        'total_tokens': int(dataset.to_pandas()['token_count'].sum())
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved successfully with {len(dataset)} samples")
    return stats


if __name__ == "__main__":
    print("Starting Qwen 2.5 Code training data preprocessing...")
    
    # 处理仓库
    dataset = process_repositories()
    
    # 保存数据集
    stats = save_dataset(dataset)
    
    print("\nPreprocessing completed!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average quality score: {stats['average_quality_score']:.2f}")