"""
恒星质量预测模型 - 简化监控系统
"""

import os
import json
import time
import logging
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import torch
import psutil
import matplotlib.pyplot as plt

# 设置字体，避免中文字体缺失问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# WandB集成（可选）
try:
    import wandb
    WANDB_AVAILABLE = True
    
    # 尝试使用环境变量中的API密钥登录
    import os
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ.get("WANDB_API_KEY"), verify=True)
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB未安装，将使用本地监控")
except Exception as e:
    print(f"WandB登录失败: {e}")
    WANDB_AVAILABLE = False


class TrainingMonitor:
    """简化的训练监控器
    
    监控训练指标和系统资源使用情况，可选择性地集成WandB
    """
    
    def __init__(self, save_dir: str = "./monitoring", log_interval: int = 10, enable_wandb: bool = False):
        """
        初始化训练监控器
        
        Args:
            save_dir: 保存监控结果的目录
            log_interval: 日志记录间隔
            enable_wandb: 是否启用WandB监控
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        
        self.log_interval = log_interval
        
        # WandB集成
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.wandb_run = None
        
        # 指标历史记录
        self.metrics_history = {
            "step": [],
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
            "learning_rate": [],
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "gpu_memory": []
        }
        
        # 系统监控器
        self.system_monitor = SystemMonitor()
        self.system_monitor.start()
    
    def initialize_wandb(self, project_name: str = "stellar-mass-prediction", config: Dict = None):
        """初始化WandB"""
        if not self.enable_wandb:
            return
            
        try:
            self.wandb_run = wandb.init(
                project=project_name,
                config=config or {},
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print(f"WandB已初始化，项目: {project_name}")
            print(f"查看实时监控: {self.wandb_run.url}")
        except Exception as e:
            print(f"WandB初始化失败: {e}")
            self.enable_wandb = False
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """
        记录训练步骤
        
        Args:
            step: 当前步骤
            metrics: 指标字典
        """
        # 更新历史记录
        self.metrics_history["step"].append(step)
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # 获取系统指标
        system_metrics = self.system_monitor.get_metrics()
        self.metrics_history["cpu_usage"].append(system_metrics["cpu_usage"])
        self.metrics_history["memory_usage"].append(system_metrics["memory_usage"])
        self.metrics_history["gpu_usage"].append(system_metrics.get("gpu_usage", 0))
        self.metrics_history["gpu_memory"].append(system_metrics.get("gpu_memory", 0))
        
        # 记录到WandB
        if self.enable_wandb and self.wandb_run:
            wandb_metrics = {}
            for key, value in metrics.items():
                wandb_metrics[key] = value
            
            # 添加系统指标
            wandb_metrics["system/cpu_usage"] = system_metrics["cpu_usage"]
            wandb_metrics["system/memory_usage"] = system_metrics["memory_usage"]
            wandb_metrics["system/gpu_usage"] = system_metrics.get("gpu_usage", 0)
            wandb_metrics["system/gpu_memory"] = system_metrics.get("gpu_memory", 0)
            
            wandb.log(wandb_metrics, step=step)
        
        # 定期保存指标
        if step % self.log_interval == 0:
            self._save_metrics()
            self._plot_training_progress()
    
    def _save_metrics(self):
        """保存指标到JSON文件"""
        metrics_file = self.save_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def _plot_training_progress(self):
        """绘制训练进度图表"""
        if len(self.metrics_history["step"]) == 0:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss
        if "train_loss" in self.metrics_history and len(self.metrics_history["train_loss"]) > 0:
            axes[0, 0].plot(self.metrics_history["step"], self.metrics_history["train_loss"], label='Train Loss')
            if "val_loss" in self.metrics_history and len(self.metrics_history["val_loss"]) > 0:
                axes[0, 0].plot(self.metrics_history["step"], self.metrics_history["val_loss"], label='Val Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # R² Score
        if "val_r2" in self.metrics_history and len(self.metrics_history["val_r2"]) > 0:
            axes[0, 1].plot(self.metrics_history["step"], self.metrics_history["val_r2"], label='Val R²')
            if "train_r2" in self.metrics_history and len(self.metrics_history["train_r2"]) > 0:
                axes[0, 1].plot(self.metrics_history["step"], self.metrics_history["train_r2"], label='Train R²')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].set_title('Training and Validation R² Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        if "learning_rate" in self.metrics_history and len(self.metrics_history["learning_rate"]) > 0:
            axes[0, 2].plot(self.metrics_history["step"], self.metrics_history["learning_rate"])
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_title('Learning Rate Variation')
            axes[0, 2].grid(True, alpha=0.3)
        
        # CPU Usage
        if len(self.metrics_history["cpu_usage"]) > 0:
            axes[1, 0].plot(self.metrics_history["step"], self.metrics_history["cpu_usage"])
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('CPU Usage (%)')
            axes[1, 0].set_title('CPU Usage')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Memory Usage
        if len(self.metrics_history["memory_usage"]) > 0:
            axes[1, 1].plot(self.metrics_history["step"], self.metrics_history["memory_usage"])
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Memory Usage (%)')
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].grid(True, alpha=0.3)
        
        # GPU Usage
        if len(self.metrics_history["gpu_usage"]) > 0 and any(self.metrics_history["gpu_usage"]):
            axes[1, 2].plot(self.metrics_history["step"], self.metrics_history["gpu_usage"])
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('GPU Usage (%)')
            axes[1, 2].set_title('GPU Usage')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "plots" / f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                    dpi=300, bbox_inches='tight')
        
        # 记录到WandB
        if self.enable_wandb and self.wandb_run:
            wandb.log({"training_progress": wandb.Image(fig)})
        
        plt.close()
    
    def finalize(self):
        """完成监控，保存最终结果"""
        self._save_metrics()
        self._plot_training_progress()
        self.system_monitor.stop()
        
        # 完成WandB运行
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.finish()


class SystemMonitor:
    """系统资源监控器
    
    监控CPU、内存和GPU使用情况
    """
    
    def __init__(self, interval: float = 1.0):
        """
        初始化系统监控器
        
        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.current_metrics = {
            "cpu_usage": 0,
            "memory_usage": 0,
            "gpu_usage": 0,
            "gpu_memory": 0
        }
    
    def start(self):
        """启动监控"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # CPU使用率
                cpu_usage = psutil.cpu_percent(interval=None)
                
                # 内存使用率
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                
                # GPU使用率和内存
                gpu_usage = 0
                gpu_memory = 0
                if torch.cuda.is_available():
                    gpu_usage = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
                
                self.current_metrics = {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "gpu_usage": gpu_usage,
                    "gpu_memory": gpu_memory
                }
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(self.interval)
    
    def get_metrics(self) -> Dict[str, float]:
        """获取当前系统指标"""
        return self.current_metrics.copy()


def setup_simple_monitoring(enable_wandb: bool = False, wandb_project: str = "stellar-mass-prediction") -> TrainingMonitor:
    """
    设置简单监控
    
    Args:
        enable_wandb: 是否启用WandB
        wandb_project: WandB项目名称
        
    Returns:
        TrainingMonitor: 训练监控器
    """
    monitor = TrainingMonitor(enable_wandb=enable_wandb)
    
    if enable_wandb:
        monitor.initialize_wandb(project_name=wandb_project)
    
    return monitor