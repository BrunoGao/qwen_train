#!/usr/bin/env python3
"""
Qwen 2.5 Code 检查点和模型保存管理器
提供智能的检查点保存、恢复和模型版本管理功能
"""

import os
import json
import shutil
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CheckpointInfo:
    """检查点信息"""
    checkpoint_dir: str
    step: int
    epoch: float
    timestamp: str
    loss: float
    model_size_gb: float
    validation_score: Optional[float] = None
    gpu_memory_usage: Optional[Dict[str, float]] = None
    training_time_hours: Optional[float] = None


@dataclass
class ModelVersion:
    """模型版本信息"""
    version: str
    checkpoint_info: CheckpointInfo
    performance_metrics: Dict[str, float]
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, base_dir: str, max_checkpoints: int = 5, save_every_n_steps: int = 1000):
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.save_every_n_steps = save_every_n_steps
        
        # 创建目录结构
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.models_dir = self.base_dir / "models" 
        self.logs_dir = self.base_dir / "logs"
        
        for dir_path in [self.checkpoints_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化状态
        self.checkpoint_history = self._load_checkpoint_history()
        self.training_start_time = time.time()
        
    def _load_checkpoint_history(self) -> List[CheckpointInfo]:
        """加载检查点历史"""
        history_file = self.base_dir / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                return [CheckpointInfo(**item) for item in data]
        return []
    
    def _save_checkpoint_history(self):
        """保存检查点历史"""
        history_file = self.base_dir / "checkpoint_history.json"
        with open(history_file, 'w') as f:
            json.dump([asdict(info) for info in self.checkpoint_history], f, indent=2)
    
    def _get_gpu_memory_usage(self) -> Dict[str, float]:
        """获取GPU内存使用情况"""
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                gpu_memory[f"gpu_{i}"] = {
                    "allocated_gb": round(memory_allocated, 2),
                    "cached_gb": round(memory_cached, 2)
                }
        return gpu_memory
    
    def _get_directory_size(self, directory: Path) -> float:
        """获取目录大小（GB）"""
        total_size = 0
        try:
            for path in directory.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        except (OSError, PermissionError):
            pass
        return total_size / 1024**3
    
    def should_save_checkpoint(self, step: int, force: bool = False) -> bool:
        """判断是否应该保存检查点"""
        if force:
            return True
        return step % self.save_every_n_steps == 0
    
    def save_checkpoint(self, 
                       model, 
                       tokenizer, 
                       optimizer, 
                       scheduler,
                       step: int, 
                       epoch: float, 
                       loss: float,
                       additional_info: Optional[Dict] = None) -> CheckpointInfo:
        """保存检查点"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint-step-{step}-epoch-{epoch:.2f}"
        checkpoint_dir = self.checkpoints_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving checkpoint to {checkpoint_dir}")
        
        # 保存模型和tokenizer
        model.save_pretrained(checkpoint_dir / "model")
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
        
        # 保存优化器和调度器状态
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'timestamp': timestamp
        }, checkpoint_dir / "training_state.pt")
        
        # 保存训练配置
        training_config = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'timestamp': timestamp,
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
        }
        
        if additional_info:
            training_config.update(additional_info)
            
        with open(checkpoint_dir / "training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)
        
        # 获取检查点信息
        checkpoint_size = self._get_directory_size(checkpoint_dir)
        gpu_memory = self._get_gpu_memory_usage()
        training_time = (time.time() - self.training_start_time) / 3600  # 小时
        
        checkpoint_info = CheckpointInfo(
            checkpoint_dir=str(checkpoint_dir),
            step=step,
            epoch=epoch,
            timestamp=timestamp,
            loss=loss,
            model_size_gb=checkpoint_size,
            gpu_memory_usage=gpu_memory,
            training_time_hours=round(training_time, 2)
        )
        
        # 更新历史记录
        self.checkpoint_history.append(checkpoint_info)
        self._save_checkpoint_history()
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        print(f"✅ Checkpoint saved: {checkpoint_size:.2f}GB, Loss: {loss:.4f}")
        return checkpoint_info
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点，保留最近的N个"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
            
        # 按步数排序，保留最新的
        self.checkpoint_history.sort(key=lambda x: x.step)
        
        # 删除最旧的检查点
        checkpoints_to_remove = self.checkpoint_history[:-self.max_checkpoints]
        
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info.checkpoint_dir)
            if checkpoint_path.exists():
                try:
                    shutil.rmtree(checkpoint_path)
                    print(f"🗑️ Removed old checkpoint: {checkpoint_path.name}")
                except Exception as e:
                    print(f"Warning: Failed to remove {checkpoint_path}: {e}")
        
        # 更新历史记录
        self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]
        self._save_checkpoint_history()
    
    def find_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """找到最新的检查点"""
        if not self.checkpoint_history:
            return None
        return max(self.checkpoint_history, key=lambda x: x.step)
    
    def load_checkpoint(self, checkpoint_info: CheckpointInfo, model, tokenizer, optimizer=None, scheduler=None):
        """加载检查点"""
        checkpoint_dir = Path(checkpoint_info.checkpoint_dir)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        print(f"Loading checkpoint from {checkpoint_dir}")
        
        # 加载模型
        model_path = checkpoint_dir / "model"
        if model_path.exists():
            model.load_state_dict(
                torch.load(model_path / "pytorch_model.bin", map_location="cpu")
            )
        
        # 加载tokenizer
        tokenizer_path = checkpoint_dir / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 加载训练状态
        training_state_path = checkpoint_dir / "training_state.pt"
        if training_state_path.exists() and optimizer is not None:
            state_dict = torch.load(training_state_path, map_location="cpu")
            
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            
            if scheduler and 'scheduler_state_dict' in state_dict and state_dict['scheduler_state_dict']:
                scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            
            return state_dict['step'], state_dict['epoch'], state_dict['loss']
        
        return checkpoint_info.step, checkpoint_info.epoch, checkpoint_info.loss
    
    def create_model_version(self, 
                           checkpoint_info: CheckpointInfo,
                           performance_metrics: Dict[str, float],
                           version_name: Optional[str] = None) -> ModelVersion:
        """创建模型版本"""
        
        if version_name is None:
            version_name = f"v1.0.{checkpoint_info.step}"
        
        # 创建版本目录
        version_dir = self.models_dir / version_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制检查点到版本目录
        checkpoint_path = Path(checkpoint_info.checkpoint_dir)
        if checkpoint_path.exists():
            shutil.copytree(checkpoint_path, version_dir / "checkpoint", dirs_exist_ok=True)
        
        # 加载配置
        config_path = checkpoint_path / "training_config.json"
        training_config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                training_config = json.load(f)
        
        model_config = training_config.get('model_config', {})
        
        # 创建版本信息
        version_info = ModelVersion(
            version=version_name,
            checkpoint_info=checkpoint_info,
            performance_metrics=performance_metrics,
            model_config=model_config,
            training_config=training_config
        )
        
        # 保存版本信息
        with open(version_dir / "version_info.json", 'w') as f:
            version_data = asdict(version_info)
            version_data['checkpoint_info'] = asdict(checkpoint_info)
            json.dump(version_data, f, indent=2)
        
        print(f"✅ Model version {version_name} created")
        return version_info
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """获取检查点摘要信息"""
        if not self.checkpoint_history:
            return {"message": "No checkpoints found"}
        
        # 基本统计
        steps = [info.step for info in self.checkpoint_history]
        losses = [info.loss for info in self.checkpoint_history]
        sizes = [info.model_size_gb for info in self.checkpoint_history]
        
        summary = {
            "total_checkpoints": len(self.checkpoint_history),
            "latest_step": max(steps),
            "step_range": {"min": min(steps), "max": max(steps)},
            "loss_range": {"min": min(losses), "max": max(losses)},
            "avg_checkpoint_size_gb": round(np.mean(sizes), 2),
            "total_storage_gb": round(sum(sizes), 2),
            "checkpoints": []
        }
        
        # 详细信息
        for info in sorted(self.checkpoint_history, key=lambda x: x.step, reverse=True):
            summary["checkpoints"].append({
                "step": info.step,
                "epoch": info.epoch,
                "loss": round(info.loss, 4),
                "size_gb": info.model_size_gb,
                "timestamp": info.timestamp,
                "training_hours": info.training_time_hours
            })
        
        return summary
    
    def export_best_model(self, 
                         output_dir: str, 
                         metric: str = "loss",
                         ascending: bool = True) -> Optional[str]:
        """导出最佳模型"""
        if not self.checkpoint_history:
            print("No checkpoints available for export")
            return None
        
        # 选择最佳检查点
        if metric == "loss":
            best_checkpoint = min(self.checkpoint_history, key=lambda x: x.loss)
        elif metric == "step":
            best_checkpoint = max(self.checkpoint_history, key=lambda x: x.step)
        else:
            best_checkpoint = self.checkpoint_history[-1]  # 最新的
        
        # 复制到输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = Path(best_checkpoint.checkpoint_dir)
        if checkpoint_path.exists():
            # 复制模型和tokenizer
            shutil.copytree(
                checkpoint_path / "model", 
                output_path / "model", 
                dirs_exist_ok=True
            )
            shutil.copytree(
                checkpoint_path / "tokenizer", 
                output_path / "tokenizer", 
                dirs_exist_ok=True
            )
            
            # 创建模型卡片
            model_card = {
                "model_name": "Qwen2.5-7B-Code-Finetuned",
                "base_model": "Qwen2.5-7B-Instruct",
                "training_info": {
                    "final_step": best_checkpoint.step,
                    "final_epoch": best_checkpoint.epoch,
                    "final_loss": best_checkpoint.loss,
                    "training_time_hours": best_checkpoint.training_time_hours
                },
                "selection_criteria": f"Best {metric}",
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(output_path / "model_card.json", 'w') as f:
                json.dump(model_card, f, indent=2)
            
            print(f"✅ Best model exported to {output_path}")
            print(f"Selection criteria: {metric} = {getattr(best_checkpoint, metric)}")
            
            return str(output_path)
        
        return None


def create_training_config_template() -> Dict[str, Any]:
    """创建训练配置模板"""
    return {
        "checkpoint_config": {
            "save_every_n_steps": 1000,
            "max_checkpoints": 5,
            "save_on_epoch_end": True,
            "save_optimizer_states": True,
            "auto_cleanup": True
        },
        
        "model_versioning": {
            "create_versions": True,
            "version_on_improvement": True,
            "improvement_threshold": 0.01,  # loss improvement
            "performance_metrics": ["loss", "perplexity", "bleu_score"]
        },
        
        "storage_management": {
            "max_storage_gb": 500,
            "compression_enabled": False,
            "remote_backup": False,
            "cleanup_policy": "keep_best_and_latest"
        },
        
        "monitoring": {
            "track_gpu_memory": True,
            "track_training_time": True,
            "log_system_metrics": True,
            "alert_on_issues": True
        }
    }


if __name__ == "__main__":
    # 示例用法
    manager = CheckpointManager("/codes/outputs", max_checkpoints=3)
    
    # 打印检查点摘要
    summary = manager.get_checkpoint_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 创建配置模板
    config_template = create_training_config_template()
    with open("/codes/checkpoint_config_template.json", 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print("✅ Checkpoint manager initialized and config template created")