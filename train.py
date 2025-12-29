# 导入必要的库
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional

# 设置matplotlib默认配置
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入模型和工具
from model import StellarTransformer, WeightedMSELoss, FocalMSELoss
from monitoring import setup_simple_monitoring, TrainingMonitor, SystemMonitor


def load_config(config_path):
    """加载配置文件（支持JSON和YAML格式）"""
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        raise


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    return device


def setup_distributed(config=None):
    """
    设置分布式训练
    
    Args:
        config: 配置对象，可选
        
    Returns:
        tuple: (rank, world_size, device)
    """
    # 默认配置
    distributed = False
    backend = 'nccl'
    
    # 从配置文件读取设置
    if config:
        distributed = config.get("system", {}).get("distributed_training", False)
        backend = config.get("system", {}).get("backend", 'nccl')
    
    if distributed and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 检查GPU可用性
        if not torch.cuda.is_available():
            raise RuntimeError("分布式训练需要CUDA支持")
        
        # 初始化分布式进程组
        torch.distributed.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        
        device = torch.device(f"cuda:{local_rank}")
        
        # 日志记录
        if rank == 0 and config and hasattr(config, 'logger'):
            config.logger.info(f"分布式训练已初始化 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}, Backend: {backend}")
        elif rank == 0:
            print(f"分布式训练已初始化 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}, Backend: {backend}")
        
        return rank, world_size, device
    else:
        # 单卡训练
        if distributed:
            if config and hasattr(config, 'logger'):
                config.logger.warning("分布式训练已启用，但未检测到必要的环境变量，将使用单卡训练")
            else:
                print("分布式训练已启用，但未检测到必要的环境变量，将使用单卡训练")
        
        # 设置设备
        device_config = config.get("system", {}).get("device", "auto") if config else "auto"
        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)
        
        return 0, 1, device


class WarmupCosineAnnealingLR(_LRScheduler):
    """预热余弦退火学习率调度器
    
    该调度器结合了预热学习率和余弦退火策略：
    1. 预热阶段：学习率从0线性增加到初始学习率
       - 自动计算预热epoch数：总epoch数的10%，最少1个epoch，最多5个epoch
    2. 余弦退火阶段：学习率从初始学习率按照余弦函数衰减到最小学习率
    
    Args:
        optimizer (torch.optim.Optimizer): 优化器
        T_max (int): 余弦退火周期（总epoch数）
        warmup_lr_start (float): 预热起始学习率
        min_lr (float): 最小学习率
        eta_min (float): 余弦退火的最小学习率（兼容PyTorch原生参数）
        last_epoch (int): 最后一个epoch索引，默认为-1
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 warmup_lr_start: float = 0.0,
                 min_lr: float = 1e-6,
                 eta_min: float = 1e-6,
                 last_epoch: int = -1):
        self.T_max = T_max
        # 自动计算预热epoch数：总epoch数的10%，最少1个epoch，最多5个epoch
        self.warmup_epochs = max(1, min(5, int(T_max * 0.1)))
        self.warmup_lr_start = warmup_lr_start
        self.min_lr = float(min_lr) if min_lr is not None else float(eta_min)
        # 确保所有初始学习率都是浮点数类型
        self.initial_lrs = [float(group['lr']) for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """计算当前epoch的学习率
        
        Returns:
            List[float]: 每个参数组的学习率
        """
        if self.last_epoch < 0:
            return self.initial_lrs
        
        # 预热阶段
        if self.last_epoch < self.warmup_epochs:
            # 线性预热：从warmup_lr_start到初始学习率
            warmup_factor = (self.last_epoch + 1) / (self.warmup_epochs + 1)
            return [self.warmup_lr_start + (base_lr - self.warmup_lr_start) * warmup_factor
                    for base_lr in self.initial_lrs]
        else:
            # 余弦退火阶段
            # 计算当前在退火周期中的位置
            progress = (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            # 余弦退火公式
            cos_factor = (1 + np.cos(np.pi * progress)) / 2
            return [self.min_lr + (base_lr - self.min_lr) * cos_factor
                    for base_lr in self.initial_lrs]
    
    def set_lr(self, optimizer: torch.optim.Optimizer, lr_values: List[float]):
        """设置优化器的学习率
        
        Args:
            optimizer (torch.optim.Optimizer): 优化器
            lr_values (List[float]): 每个参数组的学习率
        """
        for i, (param_group, lr) in enumerate(zip(optimizer.param_groups, lr_values)):
            param_group['lr'] = lr

class StellarDataset(Dataset):
    """恒星数据集"""
    def __init__(self, data, normalize=False, mean=None, std=None, y_mean=None, y_std=None):
        self.data = data
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.y_mean = y_mean
        self.y_std = y_std
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 特征：log_L, log_Teff, [Fe/H]
        features = torch.tensor([
            self.data.iloc[idx]['log_L'],
            self.data.iloc[idx]['log_Teff'],
            self.data.iloc[idx]['[Fe/H]']
        ], dtype=torch.float32)
        
        # 目标：log10_mass
        target = torch.tensor(self.data.iloc[idx]['log10_mass'], dtype=torch.float32)
        
        # 标准化处理
        if self.normalize:
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            if self.y_mean is not None and self.y_std is not None:
                target = (target - self.y_mean) / self.y_std
        
        return features, target


def calculate_r2_score(preds, targets):
    """计算R²分数"""
    return r2_score(targets.cpu().numpy(), preds.cpu().numpy())


def train_transformer(model, train_loader, val_loader, optimizer, scheduler, device, config, logger, rank=0, world_size=1):
    """
    训练Transformer模型，支持分布式训练
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        config: 配置字典
        logger: 日志记录器
        rank: 当前进程的rank
        world_size: 进程总数
        
    Returns:
        训练好的模型和训练历史
    """
    try:
        # 损失函数配置
        loss_config = config.get("training", {}).get("loss", {})
        loss_type = loss_config.get("type", "weighted_mse")
        
        # 根据配置选择损失函数
        if loss_type == "focal_mse":
            criterion = FocalMSELoss(
                alpha=loss_config.get("alpha", 0.25),
                gamma=loss_config.get("gamma", 2.0)
            )
        elif loss_type == "weighted_mse":
            criterion = WeightedMSELoss(
                low_threshold=loss_config.get("low_threshold", -1.0),
                high_threshold=loss_config.get("high_threshold", 1.0),
                low_weight=loss_config.get("low_weight", 2.0),
                high_weight=loss_config.get("high_weight", 2.0),
                mid_weight=loss_config.get("mid_weight", 1.0)
            )
        else:
            criterion = nn.MSELoss()
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'lr': []
        }
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        early_stopping_patience = config['training'].get('early_stopping_patience', 10)
        
        if rank == 0:
            logger.info(f"开始训练，共 {config['training']['num_epochs']} 轮")
            logger.info(f"早停机制已启用，耐心值: {early_stopping_patience}")
        
        for epoch in range(config['training']['num_epochs']):
            try:
                # 在分布式训练中，设置Sampler的epoch以确保数据打乱一致
                if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                # 训练阶段
                model.train()
                train_loss = 0
                batch_count = 0
                
                for batch_features, batch_targets in train_loader:
                    try:
                        batch_features = batch_features.to(device)
                        batch_targets = batch_targets.to(device)
                        
                        # 解包特征：log_L, log_Teff, [Fe/H]
                        logL = batch_features[:, 0]
                        logTeff = batch_features[:, 1]
                        FeH = batch_features[:, 2]
                        
                        # 前向传播
                        optimizer.zero_grad()
                        pred_mass = model(logL, logTeff, FeH)
                        loss = criterion(pred_mass, batch_targets)
                        
                        # 反向传播
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                        batch_count += 1
                    except Exception as e:
                        logger.error(f"批次处理失败 (epoch {epoch+1}, batch {batch_count}): {e}")
                        raise
                
                # 在分布式训练中，计算平均损失
                avg_train_loss = train_loss / len(train_loader)
                
                # 验证阶段
                model.eval()
                val_loss = 0
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        try:
                            batch_features = batch_features.to(device)
                            batch_targets = batch_targets.to(device)
                            
                            # 解包特征：log_L, log_Teff, [Fe/H]
                            logL = batch_features[:, 0]
                            logTeff = batch_features[:, 1]
                            FeH = batch_features[:, 2]
                            
                            pred_mass = model(logL, logTeff, FeH)
                            loss = criterion(pred_mass, batch_targets)
                            
                            val_loss += loss.item()
                            all_preds.append(pred_mass.cpu())
                            all_targets.append(batch_targets.cpu())
                        except Exception as e:
                            logger.error(f"验证批次处理失败 (epoch {epoch+1}): {e}")
                            raise
                
                avg_val_loss = val_loss / len(val_loader)
                
                # 计算R²
                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)
                val_r2 = calculate_r2_score(all_preds, all_targets)
                
                # 学习率调整 - 基于epoch的调度器，不需要传入验证损失
                scheduler.step()
                
                # 只在主进程保存最佳模型
                if rank == 0 and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    patience_counter = 0  # 重置耐心计数器
                    try:
                        model_save_path = Path(config["paths"]["model_dir"]) / 'best_model.pth'
                        
                        # 如果使用DDP，保存原始模型状态
                        if world_size > 1 and hasattr(model, 'module'):
                            model_state_dict = model.module.state_dict()
                        else:
                            model_state_dict = model.state_dict()
                            
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model_state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'world_size': world_size
                        }, model_save_path)
                        logger.info(f"已保存新的最佳模型到 {model_save_path} (epoch {epoch+1}, loss: {best_val_loss:.6f})")
                    except Exception as e:
                        logger.error(f"保存模型失败 (epoch {epoch+1}): {e}")
                        raise
                else:
                    patience_counter += 1  # 增加耐心计数器
                    if rank == 0:
                        logger.info(f"早停计数器: {patience_counter}/{early_stopping_patience}")
                
                # 记录历史
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['val_r2'].append(val_r2)
                history['lr'].append(optimizer.param_groups[0]['lr'])
                
                # 检查早停条件
                if patience_counter >= early_stopping_patience:
                    if rank == 0:
                        logger.info(f"早停机制触发，在epoch {epoch+1}提前终止训练")
                    break
                
                # 只在主进程打印进度
                if rank == 0 and (epoch + 1) % config['training'].get('print_every', 1) == 0:
                    logger.info(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
                    logger.info(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, R²: {val_r2:.4f}')
                    logger.info(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                    
                    # 也输出到控制台
                    print(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
                    print(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, R²: {val_r2:.4f}')
                    print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            except Exception as e:
                if rank == 0:
                    logger.error(f"训练轮次失败 (epoch {epoch+1}): {e}")
                raise
        
        if rank == 0:
            logger.info(f"训练完成！共 {config['training']['num_epochs']} 轮")
            logger.info(f"最佳验证损失: {best_val_loss:.6f} (在第 {best_epoch+1} 轮)")
            logger.info(f"最佳R²: {max(history['val_r2']):.4f}")
        
        return model, history
    except Exception as e:
        if rank == 0:
            logger.error(f"训练过程失败: {e}")
        raise


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config, logger=None, monitor=None):
    """
    训练模型
    
    Args:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        config: 配置字典
        logger: 日志记录器
        monitor: 训练监控器
    
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    # 训练参数
    num_epochs = config["training"]["num_epochs"]
    
    # 损失函数配置
    loss_config = config.get("training", {}).get("loss", {})
    loss_type = loss_config.get("type", "weighted_mse")
    
    # 根据配置选择损失函数
    if loss_type == "focal_mse":
        criterion = FocalMSELoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0)
        )
    elif loss_type == "weighted_mse":
        criterion = WeightedMSELoss(
            low_threshold=loss_config.get("low_threshold", -1.0),
            high_threshold=loss_config.get("high_threshold", 1.0),
            low_weight=loss_config.get("low_weight", 2.0),
            high_weight=loss_config.get("high_weight", 2.0),
            mid_weight=loss_config.get("mid_weight", 1.0)
        )
    else:
        criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience = config["training"].get("early_stopping_patience", 10)
    patience_counter = 0
    
    # 训练历史
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_r2": [],
        "val_r2": [],
        "learning_rate": []
    }
    
    # 系统监控器
    system_monitor = SystemMonitor() if config.get("monitoring", {}).get("enable_system_monitoring", True) else None
    if system_monitor:
        system_monitor.start()
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_targets = []
        train_predictions = []
        
        # 训练阶段
        for batch_idx, (features, targets) in enumerate(train_loader):
            # 准备数据
            logL = features[:, 0].to(device)
            logTeff = features[:, 1].to(device)
            FeH = features[:, 2].to(device)
            targets = targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(logL, logTeff, FeH)
            loss = criterion(outputs.squeeze(), targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if config["training"].get("gradient_clipping", False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"].get("max_grad_norm", 1.0))
            
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            train_targets.extend(targets.cpu().numpy())
            train_predictions.extend(outputs.detach().cpu().numpy())
            
            # 记录训练步骤
            step = epoch * len(train_loader) + batch_idx
            if step % config["logging"].get("log_interval", 10) == 0:
                # 计算R²
                train_r2 = r2_score(train_targets, train_predictions)
                
                # 获取系统指标
                system_metrics = {}
                if system_monitor:
                    system_metrics = system_monitor.get_metrics()
                
                # 记录到监控器
                metrics = {
                    "train_loss": loss.item(),
                    "train_r2": train_r2,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                
                if monitor:
                    monitor.log_step(step, metrics)
                
                # 打印日志
                logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.6f}, R²: {train_r2:.6f}")
        
        # 计算平均训练损失和R²
        avg_train_loss = train_loss / len(train_loader)
        train_r2 = r2_score(train_targets, train_predictions)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_predictions = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                logL = features[:, 0].to(device)
                logTeff = features[:, 1].to(device)
                FeH = features[:, 2].to(device)
                targets = targets.to(device)
                
                outputs = model(logL, logTeff, FeH)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                val_targets.extend(targets.cpu().numpy())
                val_predictions.extend(outputs.cpu().numpy())
        
        # 计算平均验证损失和R²
        avg_val_loss = val_loss / len(val_loader)
        val_r2 = r2_score(val_targets, val_predictions)
        
        # 更新学习率
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # 记录训练历史
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_r2"].append(train_r2)
        history["val_r2"].append(val_r2)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        
        # 获取系统指标
        system_metrics = {}
        if system_monitor:
            system_metrics = system_monitor.get_metrics()
        
        # 记录验证结果到监控器
        val_metrics = {
            "val_loss": avg_val_loss,
            "val_r2": val_r2
        }
        
        if monitor:
            monitor.log_step(epoch * len(train_loader), val_metrics)
        
        # 打印日志
        logger.info(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        logger.info(f"Epoch {epoch}, Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_val_loss,
                'config': config,
                'history': history
            }, os.path.join(config["paths"]["model_dir"], "best_model.pth"))
            
            logger.info(f"保存最佳模型，验证损失: {best_val_loss:.6f}")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            logger.info(f"早停触发，在epoch {epoch}停止训练")
            break
    
    # 停止系统监控
    if system_monitor:
        system_monitor.stop()
    
    # 完成监控
    if monitor and hasattr(monitor, 'finalize'):
        monitor.finalize()
    
    logger.info("训练完成")
    
    return model, history


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练恒星质量预测模型")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 创建文件处理器
    os.makedirs(config.get("paths", {}).get("log_dir", "./logs"), exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(config.get("paths", {}).get("log_dir", "./logs"), 
                    f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 设置监控
    enable_wandb = config.get("monitoring", {}).get("enable_wandb", False)
    monitor = setup_simple_monitoring(enable_wandb=enable_wandb)
    
    logger.info("开始训练恒星质量预测模型")
    logger.info(f"使用配置文件: {args.config}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    
    # 创建目录
    os.makedirs(config["paths"]["data_dir"], exist_ok=True)
    os.makedirs(config["paths"]["model_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    os.makedirs(config["paths"]["result_dir"], exist_ok=True)
    
    # 读取数据
    logger.info("读取数据...")
    train_data = pd.read_csv(os.path.join(config["paths"]["data_dir"], config["data"]["train_file"]), sep='\t')
    val_data = pd.read_csv(os.path.join(config["paths"]["data_dir"], config["data"]["val_file"]), sep='\t')
    
    # 创建数据集和数据加载器
    train_dataset = StellarDataset(train_data)
    val_dataset = StellarDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    logger.info(f"验证样本数: {len(val_dataset)}")
    
    # 创建模型
    model = StellarTransformer(
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"]
    ).to(device)
    
    # 记录模型信息
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 初始化监控器（如果需要）
    if monitor and hasattr(monitor, 'initialize_wandb'):
        monitor.initialize_wandb(config=config)
    
    # 创建优化器和调度器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # 始终使用预热余弦退火调度器，根据epoch自动调整学习率
    lr_scheduler_config = config["training"].get("lr_scheduler", {})
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],  # 总epoch数作为退火周期
        warmup_lr_start=0.0,  # 固定从0开始预热
        min_lr=lr_scheduler_config.get("min_lr", 1e-6)  # 最小学习率
    )
    logger.info(f"使用预热余弦退火调度器，自动计算预热{max(1, min(5, int(config['training']['num_epochs'] * 0.1)))}个epoch，最小学习率{lr_scheduler_config.get('min_lr', 1e-6)}")
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"从epoch {start_epoch}恢复训练")
    
    # 训练模型
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        monitor=monitor
    )
    
    # 保存训练历史
    history_path = os.path.join(config["paths"]["result_dir"], "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"训练历史已保存到: {history_path}")
    
    logger.info("训练完成")


if __name__ == "__main__":
    main()