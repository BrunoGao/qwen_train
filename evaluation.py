#!/usr/bin/env python3
"""
Qwen 2.5 Code 模型评估框架
包含代码理解、生成、补全等多个维度的评估
"""

import os
import json
import time
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """评估结果数据结构"""
    task: str
    score: float
    details: Dict[str, Any]
    examples: List[Dict[str, Any]]
    timestamp: str


class CodeEvaluator:
    """代码模型评估器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """初始化评估器"""
        print(f"Loading model from {model_path}...")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str, max_length: int = 2048, temperature: float = 0.3) -> str:
        """生成模型响应"""
        messages = [
            {"role": "system", "content": "你是一个代码分析专家，专门帮助用户理解、分析和优化代码。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用Qwen的chat template
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 提取生成的部分
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()


class CodeUnderstandingEvaluator:
    """代码理解能力评估"""
    
    def __init__(self, evaluator: CodeEvaluator):
        self.evaluator = evaluator
        
        # 代码理解测试用例
        self.test_cases = [
            {
                "language": "Python",
                "code": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""",
                "question": "请解释这段代码的功能和算法原理",
                "expected_keywords": ["快速排序", "递归", "分治", "pivot", "时间复杂度"]
            },
            {
                "language": "Java",
                "code": """
public class Singleton {
    private static volatile Singleton instance;
    private Singleton() {}
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
""",
                "question": "分析这个设计模式的实现和线程安全性",
                "expected_keywords": ["单例模式", "双重检查", "volatile", "线程安全", "懒加载"]
            },
            {
                "language": "Go",
                "code": """
package main

import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {
    defer wg.Done()
    for j := range jobs {
        fmt.Printf("worker %d processing job %d\\n", id, j)
        time.Sleep(time.Second)
        results <- j * 2
    }
}

func main() {
    jobs := make(chan int, 100)
    results := make(chan int, 100)
    var wg sync.WaitGroup

    for w := 1; w <= 3; w++ {
        wg.Add(1)
        go worker(w, jobs, results, &wg)
    }

    for j := 1; j <= 9; j++ {
        jobs <- j
    }
    close(jobs)

    wg.Wait()
    close(results)

    for r := range results {
        fmt.Println("result:", r)
    }
}
""",
                "question": "解释这个并发程序的工作原理",
                "expected_keywords": ["goroutine", "channel", "并发", "工作池", "同步"]
            }
        ]
    
    def evaluate(self) -> EvaluationResult:
        """评估代码理解能力"""
        print("Evaluating code understanding...")
        
        total_score = 0
        examples = []
        
        for i, test_case in enumerate(tqdm(self.test_cases, desc="Code Understanding")):
            prompt = f"请分析以下{test_case['language']}代码：\n\n```{test_case['language'].lower()}\n{test_case['code']}\n```\n\n{test_case['question']}"
            
            try:
                response = self.evaluator.generate_response(prompt, max_length=1024)
                
                # 评估响应质量
                score = self._score_response(response, test_case['expected_keywords'])
                total_score += score
                
                examples.append({
                    "test_case": i + 1,
                    "language": test_case['language'],
                    "prompt": prompt,
                    "response": response,
                    "score": score,
                    "expected_keywords": test_case['expected_keywords']
                })
                
            except Exception as e:
                print(f"Error in test case {i+1}: {e}")
                examples.append({
                    "test_case": i + 1,
                    "error": str(e),
                    "score": 0
                })
        
        avg_score = total_score / len(self.test_cases)
        
        return EvaluationResult(
            task="code_understanding",
            score=avg_score,
            details={
                "total_cases": len(self.test_cases),
                "average_score": avg_score,
                "score_distribution": [ex.get("score", 0) for ex in examples]
            },
            examples=examples,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _score_response(self, response: str, expected_keywords: List[str]) -> float:
        """评分响应质量"""
        response_lower = response.lower()
        
        # 关键词匹配度
        keyword_score = sum(1 for kw in expected_keywords if kw.lower() in response_lower) / len(expected_keywords)
        
        # 长度合理性 (50-500字符)
        length_score = min(1.0, max(0.1, len(response) / 500))
        
        # 结构化程度 (包含技术术语)
        tech_terms = ["函数", "方法", "类", "变量", "算法", "复杂度", "设计模式", "并发", "同步"]
        structure_score = min(1.0, sum(1 for term in tech_terms if term in response) / 3)
        
        # 综合评分
        final_score = (keyword_score * 0.5 + length_score * 0.2 + structure_score * 0.3)
        return round(final_score, 2)


class CodeGenerationEvaluator:
    """代码生成能力评估"""
    
    def __init__(self, evaluator: CodeEvaluator):
        self.evaluator = evaluator
        
        self.test_cases = [
            {
                "language": "Python",
                "task": "实现一个二叉搜索树的插入和搜索方法",
                "expected_features": ["class", "def insert", "def search", "self.left", "self.right"]
            },
            {
                "language": "Java", 
                "task": "创建一个线程安全的计数器类",
                "expected_features": ["class", "synchronized", "private", "public", "volatile"]
            },
            {
                "language": "Go",
                "task": "实现一个HTTP客户端包装器，支持重试机制",
                "expected_features": ["func", "http.Client", "retry", "time.Sleep", "error"]
            },
            {
                "language": "Rust",
                "task": "编写一个安全的并发计数器",
                "expected_features": ["struct", "Mutex", "Arc", "impl", "unsafe"]
            }
        ]
    
    def evaluate(self) -> EvaluationResult:
        """评估代码生成能力"""
        print("Evaluating code generation...")
        
        total_score = 0
        examples = []
        
        for i, test_case in enumerate(tqdm(self.test_cases, desc="Code Generation")):
            prompt = f"请用{test_case['language']}语言{test_case['task']}。要求代码清晰、完整、可执行。"
            
            try:
                response = self.evaluator.generate_response(prompt, max_length=1536)
                
                # 评估生成的代码
                score = self._score_code_generation(response, test_case['expected_features'], test_case['language'])
                total_score += score
                
                examples.append({
                    "test_case": i + 1,
                    "language": test_case['language'],
                    "task": test_case['task'],
                    "generated_code": response,
                    "score": score,
                    "expected_features": test_case['expected_features']
                })
                
            except Exception as e:
                print(f"Error in generation test case {i+1}: {e}")
                examples.append({
                    "test_case": i + 1,
                    "error": str(e),
                    "score": 0
                })
        
        avg_score = total_score / len(self.test_cases)
        
        return EvaluationResult(
            task="code_generation",
            score=avg_score,
            details={
                "total_cases": len(self.test_cases),
                "average_score": avg_score,
                "languages": [tc['language'] for tc in self.test_cases]
            },
            examples=examples,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _score_code_generation(self, code: str, expected_features: List[str], language: str) -> float:
        """评分生成的代码"""
        # 特征匹配度
        feature_score = sum(1 for feature in expected_features if feature in code) / len(expected_features)
        
        # 代码块检测
        has_code_block = "```" in code or any(
            keyword in code for keyword in ["def ", "class ", "func ", "public ", "private "]
        )
        code_block_score = 1.0 if has_code_block else 0.3
        
        # 语法结构检测
        syntax_patterns = {
            "Python": ["def ", "class ", "if ", "for ", "import "],
            "Java": ["public ", "class ", "private ", "import ", "{"],
            "Go": ["func ", "package ", "import ", "type ", "var "],
            "Rust": ["fn ", "struct ", "impl ", "use ", "let "]
        }
        
        patterns = syntax_patterns.get(language, [])
        syntax_score = min(1.0, sum(1 for pattern in patterns if pattern in code) / 3)
        
        # 综合评分
        final_score = (feature_score * 0.4 + code_block_score * 0.3 + syntax_score * 0.3)
        return round(final_score, 2)


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self, evaluator: CodeEvaluator):
        self.evaluator = evaluator
    
    def evaluate_latency(self, num_samples: int = 10) -> Dict[str, float]:
        """评估推理延迟"""
        print(f"Evaluating inference latency with {num_samples} samples...")
        
        test_prompt = "请解释以下Python代码的功能：\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
        
        latencies = []
        
        for i in tqdm(range(num_samples), desc="Latency Test"):
            start_time = time.time()
            response = self.evaluator.generate_response(test_prompt, max_length=512)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        return {
            "mean_latency": np.mean(latencies),
            "median_latency": np.median(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "samples": num_samples
        }
    
    def evaluate_throughput(self, batch_sizes: List[int] = [1, 2, 4]) -> Dict[str, Any]:
        """评估吞吐量"""
        print("Evaluating throughput...")
        
        results = {}
        test_prompts = [
            "解释这个算法的时间复杂度",
            "优化这段代码的性能",
            "分析这个设计模式的优缺点"
        ] * 10  # 30个prompts
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            start_time = time.time()
            
            # 批量处理
            for i in range(0, min(len(test_prompts), 10), batch_size):
                batch = test_prompts[i:i+batch_size]
                for prompt in batch:
                    response = self.evaluator.generate_response(prompt, max_length=256)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            results[f"batch_{batch_size}"] = {
                "total_time": total_time,
                "samples_processed": min(len(test_prompts), 10),
                "samples_per_second": min(len(test_prompts), 10) / total_time
            }
        
        return results


def run_comprehensive_evaluation(model_path: str, output_dir: str = "/codes/evaluation_results"):
    """运行全面评估"""
    print("🧪 Starting comprehensive evaluation...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化评估器
    evaluator = CodeEvaluator(model_path)
    
    # 运行各项评估
    results = {}
    
    # 1. 代码理解评估
    understanding_evaluator = CodeUnderstandingEvaluator(evaluator)
    results['code_understanding'] = understanding_evaluator.evaluate()
    
    # 2. 代码生成评估
    generation_evaluator = CodeGenerationEvaluator(evaluator)
    results['code_generation'] = generation_evaluator.evaluate()
    
    # 3. 性能评估
    performance_evaluator = PerformanceEvaluator(evaluator)
    results['performance'] = {
        'latency': performance_evaluator.evaluate_latency(),
        'throughput': performance_evaluator.evaluate_throughput()
    }
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = Path(output_dir) / f"evaluation_results_{timestamp}.json"
    
    # 转换结果为可序列化格式
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, EvaluationResult):
            serializable_results[key] = {
                'task': value.task,
                'score': value.score,
                'details': value.details,
                'examples': value.examples,
                'timestamp': value.timestamp
            }
        else:
            serializable_results[key] = value
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n📊 Evaluation Summary:")
    print(f"Code Understanding: {results['code_understanding'].score:.2f}")
    print(f"Code Generation: {results['code_generation'].score:.2f}")
    print(f"Avg Latency: {results['performance']['latency']['mean_latency']:.2f}s")
    print(f"Results saved to: {result_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Qwen Code model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="/codes/evaluation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    run_comprehensive_evaluation(args.model_path, args.output_dir)