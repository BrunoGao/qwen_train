#!/usr/bin/env python3
"""
双A40训练监控和检查点管理器
专门针对双GPU环境优化的监控系统
"""

import os
import json
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
import queue
import signal

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DualA40Metrics:
    """双A40监控指标"""
    timestamp: str
    step: int
    epoch: float
    loss: float
    learning_rate: float
    gpu_0_memory_gb: float
    gpu_1_memory_gb: float
    gpu_0_utilization: int
    gpu_1_utilization: int
    gpu_0_temperature: int
    gpu_1_temperature: int
    cpu_percent: float
    memory_percent: float
    throughput_samples_per_sec: float
    gradient_norm: float
    eta_hours: float


class DualA40Monitor:
    """双A40训练监控器"""
    
    def __init__(self, output_dir: str, checkpoint_interval: int = 1000):
        self.output_dir = Path(output_dir)
        self.checkpoint_interval = checkpoint_interval
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 监控文件
        self.metrics_file = self.output_dir / "training_metrics.jsonl"
        self.alerts_file = self.output_dir / "alerts.log"
        
        # 监控状态
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.start_time = time.time()
        self.last_checkpoint_time = time.time()
        
        # 性能历史
        self.step_history = []
        self.loss_history = []
        self.memory_history = []
        
        # 阈值设置
        self.memory_threshold = 45  # GB
        self.temperature_threshold = 80  # 摄氏度
        self.utilization_min = 70  # 最小GPU利用率
        
        print(f"📊 Dual A40 Monitor initialized")
        print(f"📁 Output: {output_dir}")

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """获取GPU指标"""
        gpu_metrics = {}
        
        try:
            # 使用nvidia-ml-py或nvidia-smi
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines[:2]):  # 只取前两个GPU
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_metrics[f'gpu_{i}'] = {
                            'memory_used_mb': int(parts[1]),
                            'memory_total_mb': int(parts[2]),
                            'memory_used_gb': round(int(parts[1]) / 1024, 2),
                            'utilization_percent': int(parts[3]),
                            'temperature_celsius': int(parts[4])
                        }
        except Exception as e:
            print(f"⚠️ GPU metrics collection failed: {e}")
        
        # 备用方法：使用PyTorch
        if not gpu_metrics and torch.cuda.is_available():
            for i in range(min(torch.cuda.device_count(), 2)):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    gpu_metrics[f'gpu_{i}'] = {
                        'memory_used_gb': round(memory_allocated / 1024**3, 2),
                        'memory_reserved_gb': round(memory_reserved / 1024**3, 2),
                        'utilization_percent': 0,  # 无法通过PyTorch获取
                        'temperature_celsius': 0   # 无法通过PyTorch获取
                    }
                except Exception:
                    pass
        
        return gpu_metrics

    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': round(cpu_percent, 1),
                'memory_percent': round(memory.percent, 1),
                'memory_used_gb': round(memory.used / 1024**3, 2),
                'memory_total_gb': round(memory.total / 1024**3, 2)
            }
        except Exception as e:
            print(f"⚠️ System metrics collection failed: {e}")
            return {}

    def calculate_throughput(self, current_step: int) -> float:
        """计算训练吞吐量"""
        if len(self.step_history) < 2:
            return 0.0
        
        # 使用最近10个步骤计算平均速度
        recent_steps = self.step_history[-10:]
        if len(recent_steps) >= 2:
            time_diff = recent_steps[-1]['time'] - recent_steps[0]['time']
            step_diff = recent_steps[-1]['step'] - recent_steps[0]['step']
            
            if time_diff > 0:
                return round(step_diff / time_diff, 4)
        
        return 0.0

    def record_metrics(self, training_metrics: Dict[str, Any]):
        """记录训练指标"""
        current_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # 获取系统指标
        gpu_metrics = self.get_gpu_metrics()
        system_metrics = self.get_system_metrics()
        
        # 更新历史记录
        step = training_metrics.get('step', 0)
        loss = training_metrics.get('loss', 0.0)
        
        self.step_history.append({'time': current_time, 'step': step})
        self.loss_history.append({'time': current_time, 'loss': loss})
        
        # 保持历史记录在合理大小
        if len(self.step_history) > 100:
            self.step_history = self.step_history[-50:]
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-50:]
        
        # 计算吞吐量
        throughput = self.calculate_throughput(step)
        
        # 估算剩余时间
        eta_hours = 0.0
        if throughput > 0 and 'max_steps' in training_metrics:
            remaining_steps = training_metrics['max_steps'] - step
            eta_seconds = remaining_steps / throughput
            eta_hours = round(eta_seconds / 3600, 2)
        
        # 构建完整指标
        metrics = DualA40Metrics(
            timestamp=timestamp,
            step=step,
            epoch=training_metrics.get('epoch', 0.0),
            loss=loss,
            learning_rate=training_metrics.get('learning_rate', 0.0),
            gpu_0_memory_gb=gpu_metrics.get('gpu_0', {}).get('memory_used_gb', 0.0),
            gpu_1_memory_gb=gpu_metrics.get('gpu_1', {}).get('memory_used_gb', 0.0),
            gpu_0_utilization=gpu_metrics.get('gpu_0', {}).get('utilization_percent', 0),
            gpu_1_utilization=gpu_metrics.get('gpu_1', {}).get('utilization_percent', 0),
            gpu_0_temperature=gpu_metrics.get('gpu_0', {}).get('temperature_celsius', 0),
            gpu_1_temperature=gpu_metrics.get('gpu_1', {}).get('temperature_celsius', 0),
            cpu_percent=system_metrics.get('cpu_percent', 0.0),
            memory_percent=system_metrics.get('memory_percent', 0.0),
            throughput_samples_per_sec=throughput,
            gradient_norm=training_metrics.get('gradient_norm', 0.0),
            eta_hours=eta_hours
        )
        
        # 保存指标
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
        
        # 检查异常
        self.check_alerts(metrics, gpu_metrics, system_metrics)
        
        # 打印进度
        self.print_progress(metrics)
        
        return metrics

    def check_alerts(self, metrics: DualA40Metrics, gpu_metrics: Dict, system_metrics: Dict):
        """检查异常情况并记录警报"""
        alerts = []
        
        # GPU内存检查
        if metrics.gpu_0_memory_gb > self.memory_threshold:
            alerts.append(f"GPU 0 memory high: {metrics.gpu_0_memory_gb:.1f}GB")
        if metrics.gpu_1_memory_gb > self.memory_threshold:
            alerts.append(f"GPU 1 memory high: {metrics.gpu_1_memory_gb:.1f}GB")
        
        # GPU温度检查
        if metrics.gpu_0_temperature > self.temperature_threshold:
            alerts.append(f"GPU 0 temperature high: {metrics.gpu_0_temperature}°C")
        if metrics.gpu_1_temperature > self.temperature_threshold:
            alerts.append(f"GPU 1 temperature high: {metrics.gpu_1_temperature}°C")
        
        # GPU利用率检查
        if metrics.gpu_0_utilization < self.utilization_min and metrics.gpu_0_utilization > 0:
            alerts.append(f"GPU 0 utilization low: {metrics.gpu_0_utilization}%")
        if metrics.gpu_1_utilization < self.utilization_min and metrics.gpu_1_utilization > 0:
            alerts.append(f"GPU 1 utilization low: {metrics.gpu_1_utilization}%")
        
        # 系统内存检查
        if metrics.memory_percent > 90:
            alerts.append(f"System memory high: {metrics.memory_percent:.1f}%")
        
        # Loss检查
        if len(self.loss_history) >= 10:
            recent_losses = [h['loss'] for h in self.loss_history[-10:]]
            if all(loss > recent_losses[0] for loss in recent_losses[5:]):
                alerts.append("Loss not decreasing in recent steps")
        
        # 记录警报
        if alerts:
            timestamp = datetime.now().isoformat()
            with open(self.alerts_file, 'a') as f:
                for alert in alerts:
                    f.write(f"[{timestamp}] ⚠️ {alert}\n")
                    print(f"⚠️ ALERT: {alert}")

    def print_progress(self, metrics: DualA40Metrics):
        """打印训练进度"""
        if metrics.step % 50 == 0:  # 每50步打印一次
            gpu_mem = f"GPU: {metrics.gpu_0_memory_gb:.1f}GB/{metrics.gpu_1_memory_gb:.1f}GB"
            gpu_util = f"Util: {metrics.gpu_0_utilization}%/{metrics.gpu_1_utilization}%"
            
            print(f"Step {metrics.step} | Loss: {metrics.loss:.4f} | {gpu_mem} | {gpu_util} | "
                  f"Speed: {metrics.throughput_samples_per_sec:.3f} steps/s | ETA: {metrics.eta_hours:.1f}h")

    def should_save_checkpoint(self, step: int, force: bool = False) -> bool:
        """判断是否应该保存检查点"""
        if force:
            return True
        
        # 按步数间隔
        if step % self.checkpoint_interval == 0:
            return True
        
        # 按时间间隔 (每小时)
        current_time = time.time()
        if current_time - self.last_checkpoint_time > 3600:
            self.last_checkpoint_time = current_time
            return True
        
        return False

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.step_history or not self.loss_history:
            return {"message": "No training data available"}
        
        current_time = time.time()
        training_time = current_time - self.start_time
        
        # 基本统计
        steps = [h['step'] for h in self.step_history]
        losses = [h['loss'] for h in self.loss_history]
        
        summary = {
            "training_time_hours": round(training_time / 3600, 2),
            "total_steps": max(steps) if steps else 0,
            "current_loss": losses[-1] if losses else 0,
            "best_loss": min(losses) if losses else 0,
            "avg_throughput": self.calculate_throughput(max(steps) if steps else 0),
            "gpu_memory_usage": {
                "max_gpu_0": max([float(line.split()[12]) for line in open(self.metrics_file, 'r') 
                                if line.strip() and 'gpu_0_memory_gb' in line] or [0]),
                "max_gpu_1": max([float(line.split()[14]) for line in open(self.metrics_file, 'r') 
                                if line.strip() and 'gpu_1_memory_gb' in line] or [0])
            }
        }
        
        return summary

    def cleanup_old_metrics(self, keep_hours: int = 48):
        """清理旧的监控数据"""
        if not self.metrics_file.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(hours=keep_hours)
        
        # 读取现有数据
        valid_lines = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if timestamp > cutoff_time:
                        valid_lines.append(line)
                except:
                    continue
        
        # 重写文件
        with open(self.metrics_file, 'w') as f:
            f.writelines(valid_lines)
        
        print(f"🗑️ Cleaned old metrics, kept {len(valid_lines)} records")


def create_dual_a40_monitoring_dashboard():
    """创建双A40监控仪表板HTML"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dual A40 Training Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-box { 
            display: inline-block; 
            background: #f5f5f5; 
            padding: 15px; 
            margin: 10px; 
            border-radius: 5px; 
            min-width: 150px;
        }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 14px; color: #7f8c8d; }
        .alert { background: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; }
        .chart { width: 100%; height: 400px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>🚀 Qwen 2.5 7B Code - Dual A40 Training Monitor</h1>
    
    <div id="metrics">
        <div class="metric-box">
            <div class="metric-value" id="current-loss">--</div>
            <div class="metric-label">Current Loss</div>
        </div>
        <div class="metric-box">
            <div class="metric-value" id="gpu-0-memory">--</div>
            <div class="metric-label">GPU 0 Memory (GB)</div>
        </div>
        <div class="metric-box">
            <div class="metric-value" id="gpu-1-memory">--</div>
            <div class="metric-label">GPU 1 Memory (GB)</div>
        </div>
        <div class="metric-box">
            <div class="metric-value" id="throughput">--</div>
            <div class="metric-label">Steps/sec</div>
        </div>
        <div class="metric-box">
            <div class="metric-value" id="eta">--</div>
            <div class="metric-label">ETA (hours)</div>
        </div>
    </div>
    
    <div id="alerts"></div>
    
    <div id="loss-chart" class="chart"></div>
    <div id="memory-chart" class="chart"></div>
    <div id="gpu-util-chart" class="chart"></div>
    
    <script>
        // 模拟数据更新
        function updateDashboard() {
            // 这里应该从实际的metrics文件读取数据
            console.log('Updating dashboard...');
        }
        
        // 每30秒更新一次
        setInterval(updateDashboard, 30000);
        updateDashboard();
    </script>
</body>
</html>
"""
    
    dashboard_path = "/codes/dual_a40_dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"📊 Dashboard created: {dashboard_path}")
    return dashboard_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual A40 Training Monitor")
    parser.add_argument("--output_dir", default="/codes/outputs", help="Output directory")
    parser.add_argument("--create_dashboard", action="store_true", help="Create HTML dashboard")
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = DualA40Monitor(args.output_dir)
    
    if args.create_dashboard:
        create_dual_a40_monitoring_dashboard()
    
    # 示例：模拟训练指标更新
    print("🔄 Monitor ready. Press Ctrl+C to stop.")
    
    try:
        step = 0
        while True:
            # 模拟训练指标
            training_metrics = {
                'step': step,
                'epoch': step / 1000,
                'loss': 3.0 - (step * 0.001),
                'learning_rate': 1.5e-5,
                'gradient_norm': 0.8,
                'max_steps': 50000
            }
            
            metrics = monitor.record_metrics(training_metrics)
            
            step += 1
            time.sleep(10)  # 每10秒更新一次
            
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
        summary = monitor.get_training_summary()
        print("\n📊 Training Summary:")
        print(json.dumps(summary, indent=2))