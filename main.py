import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
import os
import traceback
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import StellarTransformerWithPooling
from train import StellarDataset, train_transformer, WarmupCosineAnnealingLR


def load_config(config_path: str = "config.yaml"):
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


def setup_logging(config):
    """设置日志记录"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("paths", {}).get("log_file", "training.log")
    
    # 创建日志目录（如果路径包含目录）
    log_dir = os.path.dirname(log_file)
    if log_dir:  # 只有当目录名不为空时才创建
        os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 创建logger实例
    logger = logging.getLogger(__name__)
    logger.info("配置加载完成")
    return logger


def setup_device(config):
    """
    设置计算设备，支持分布式训练
    """
    import torch
    # 检查是否启用分布式训练
    distributed = config.get("system", {}).get("distributed_training", False)
    
    if distributed:
        try:
            import torch.distributed as dist
            import torch.utils.data.distributed
            
            # 打印环境变量信息（用于调试）
            print(f"环境变量检查 - RANK: {os.environ.get('RANK')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
            
            # 检查环境变量是否存在
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
                rank = int(os.environ['RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
                local_rank = int(os.environ['LOCAL_RANK'])
                
                print(f"进程 {rank} 准备初始化分布式进程组...")
                
                # 初始化分布式进程组
                dist.init_process_group(backend='nccl')
                torch.cuda.set_device(local_rank)
                
                device = torch.device(f"cuda:{local_rank}")
                config['rank'] = rank
                config['world_size'] = world_size
                config['local_rank'] = local_rank
                config['device'] = device
                
                # 设置全局rank和world_size
                os.environ['RANK'] = str(rank)
                os.environ['WORLD_SIZE'] = str(world_size)
                os.environ['LOCAL_RANK'] = str(local_rank)
                
                # 只在主进程设置日志
                if rank == 0:
                    config['logger'].info(f"分布式训练已初始化 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
            else:
                # 如果没有设置环境变量，回退到单卡训练模式
                print("警告: 未设置分布式训练环境变量，回退到单卡训练模式")
                distributed = False
        except Exception as e:
            print(f"分布式训练初始化失败: {e}，回退到单卡训练模式")
            import traceback
            traceback.print_exc()
            distributed = False
    
    # 单卡训练逻辑（如果分布式训练未成功初始化）
    if not distributed:
        # 更新配置，确保distributed_training为False
        if not config.get("system"):
            config["system"] = {}
        config["system"]["distributed_training"] = False
        
        device_config = config.get("system", {}).get("device", "auto")
        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)
        
        # 设置默认值
        config['rank'] = 0
        config['device'] = device
        config['world_size'] = 1
        config['local_rank'] = 0
        
        config['logger'].info(f"使用设备: {device}")
    
    return device


def load_data(config):
    """
    加载预处理好的数据集
    
    Args:
        config: 配置对象
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    data_dir = Path(config['paths']['data_dir'])
    feature_columns = config['data']['feature_columns']
    train_file = config['data']['train_file']
    val_file = config['data']['val_file']
    test_file = config['data']['test_file']
    
    config['logger'].info(f"从 {data_dir} 加载数据...")
    config['logger'].info(f"特征顺序: {feature_columns}")
    
    try:
        train_df = pd.read_csv(data_dir / train_file, sep='\t')
        val_df = pd.read_csv(data_dir / val_file, sep='\t')
        test_df = pd.read_csv(data_dir / test_file, sep='\t')
        config['logger'].info(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}, 测试集大小: {len(test_df)}")
        return train_df, val_df, test_df
    except FileNotFoundError as e:
        config['logger'].error(f"文件读取失败: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        config['logger'].error(f"数据文件为空: {e}")
        raise
    except Exception as e:
        config['logger'].error(f"数据加载失败: {e}")
        raise


def extract_features(config, dfs):
    """
    从数据框中提取特征和目标变量，并计算标准化参数
    
    Args:
        config: 配置对象
        dfs (list): 数据框列表 [train_df, val_df, test_df]
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, mean, std, y_mean, y_std)
    """
    feature_columns = config['data']['feature_columns']
    target_column = config['data']['target_column']
    
    try:
        # 验证特征列是否存在
        for col in feature_columns + [target_column]:
            if col not in dfs[0].columns:
                raise ValueError(f"列 {col} 不在训练数据中")
        
        train_df, val_df, test_df = dfs
        X_train = train_df[feature_columns].values
        y_train = train_df[target_column].values
        
        X_val = val_df[feature_columns].values
        y_val = val_df[target_column].values
        
        X_test = test_df[feature_columns].values
        y_test = test_df[target_column].values
        
        # 验证数据完整性
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            config['logger'].warning("训练数据中存在NaN或无穷大值")
        
        # 计算标准化参数（仅使用训练集）
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        y_mean = y_train.mean()
        y_std = y_train.std()
        
        # 保存标准化参数到配置中，便于后续使用
        config['standardization'] = {
            'mean': mean,
            'std': std,
            'y_mean': y_mean,
            'y_std': y_std
        }
        
        config['logger'].info("数据加载和预处理完成")
        config['logger'].info(f"标准化参数 - 特征均值: {mean}, 特征标准差: {std}")
        config['logger'].info(f"标准化参数 - 目标均值: {y_mean}, 目标标准差: {y_std}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, mean, std, y_mean, y_std
    except ValueError as e:
        config['logger'].error(f"数据验证失败: {e}")
        raise
    except Exception as e:
        config['logger'].error(f"特征提取失败: {e}")
        raise


def create_data_loaders(config, X_train, X_val, X_test, y_train, y_val, y_test, mean=None, std=None, y_mean=None, y_std=None):
    """
    创建数据加载器，支持分布式训练和数据标准化
    
    Args:
        config: 配置对象
        X_train, X_val, X_test: 特征数据
        y_train, y_val, y_test: 目标数据
        mean, std: 特征标准化参数（均值和标准差）
        y_mean, y_std: 目标变量标准化参数（均值和标准差）
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    try:
        from train import StellarDataset
        
        # 将X和y合并成DataFrame，以符合StellarDataset的接口要求
        feature_columns = config['data']['feature_columns']
        target_column = config['data']['target_column']
        
        # 创建训练数据集
        train_df = pd.DataFrame(X_train, columns=feature_columns)
        train_df[target_column] = y_train
        train_dataset = StellarDataset(
            train_df, 
            normalize=True, 
            mean=torch.tensor(mean, dtype=torch.float32) if mean is not None else None,
            std=torch.tensor(std, dtype=torch.float32) if std is not None else None,
            y_mean=torch.tensor(y_mean, dtype=torch.float32) if y_mean is not None else None,
            y_std=torch.tensor(y_std, dtype=torch.float32) if y_std is not None else None
        )
        
        # 创建验证数据集
        val_df = pd.DataFrame(X_val, columns=feature_columns)
        val_df[target_column] = y_val
        val_dataset = StellarDataset(
            val_df, 
            normalize=True, 
            mean=torch.tensor(mean, dtype=torch.float32) if mean is not None else None,
            std=torch.tensor(std, dtype=torch.float32) if std is not None else None,
            y_mean=torch.tensor(y_mean, dtype=torch.float32) if y_mean is not None else None,
            y_std=torch.tensor(y_std, dtype=torch.float32) if y_std is not None else None
        )
        
        # 创建测试数据集
        test_df = pd.DataFrame(X_test, columns=feature_columns)
        test_df[target_column] = y_test
        test_dataset = StellarDataset(
            test_df, 
            normalize=True, 
            mean=torch.tensor(mean, dtype=torch.float32) if mean is not None else None,
            std=torch.tensor(std, dtype=torch.float32) if std is not None else None,
            y_mean=torch.tensor(y_mean, dtype=torch.float32) if y_mean is not None else None,
            y_std=torch.tensor(y_std, dtype=torch.float32) if y_std is not None else None
        )
        
        batch_size = config['training']['batch_size']
        num_workers = config['training']['num_workers']
        
        if config['rank'] == 0:
            config['logger'].info(f"使用批次大小: {batch_size}")
        
        # 检查是否启用分布式训练
        distributed = config.get("system", {}).get("distributed_training", False)
        
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            
            # 为训练集创建DistributedSampler
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=config['world_size'],
                rank=config['rank'],
                shuffle=True  # 分布式训练中，由sampler处理shuffle
            )
            
            # 验证集和测试集不需要shuffle
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=config['world_size'],
                rank=config['rank'],
                shuffle=False
            )
            
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=config['world_size'],
                rank=config['rank'],
                shuffle=False
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,  # 使用sampler替代shuffle
                num_workers=num_workers
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                num_workers=num_workers
            )
        else:
            # 单卡训练
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        
        if config['rank'] == 0:
            config['logger'].info("数据加载器创建完成")
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        if config['rank'] == 0:
            config['logger'].error(f"数据加载器创建失败: {e}")
        raise


def initialize_model(config):
    """
    初始化模型，支持分布式训练
    
    Args:
        config: 配置对象
        
    Returns:
        torch.nn.Module: 初始化的模型（可能是DDP包装的）
    """
    try:
        from model import StellarTransformer, StellarTransformerWithPooling
        
        config['logger'].info(f"使用模型: StellarTransformerWithPooling")
        model = StellarTransformerWithPooling(**config['model'])
        
        model.to(config['device'])
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 检查是否启用分布式训练
        distributed = config.get("system", {}).get("distributed_training", False)
        
        if distributed:
            import torch.nn.parallel as nn_parallel
            
            # 使用DDP包装模型
            model = nn_parallel.DistributedDataParallel(
                model,
                device_ids=[config['local_rank']],
                output_device=config['local_rank']
            )
            
            if config['rank'] == 0:
                config['logger'].info(f"模型已使用DDP包装")
        
        if config['rank'] == 0:
            config['logger'].info(f"模型参数总数: {total_params:,}")
            config['logger'].info(f"模型已移动到 {config['device']}")
        
        return model
    except ImportError as e:
        if config['rank'] == 0:
            config['logger'].error(f"导入模型失败: {e}")
        raise
    except ValueError as e:
        if config['rank'] == 0:
            config['logger'].error(f"模型初始化失败: {e}")
        raise
    except Exception as e:
        if config['rank'] == 0:
            config['logger'].error(f"模型创建失败: {e}")
        raise


def save_training_history(config, history):
    """
    保存训练历史
    
    Args:
        config: 配置对象
        history (dict): 训练历史字典
    """
    try:
        history_save_path = Path(config["paths"]["model_dir"]) / "training_history.json"
        with open(history_save_path, 'w') as f:
            # 转换numpy类型为python类型
            history_serializable = {}
            for key, value in history.items():
                if isinstance(value, list):
                    history_serializable[key] = [float(v) for v in value]
                else:
                    history_serializable[key] = float(value)
            json.dump(history_serializable, f, indent=4)
        config['logger'].info(f"训练历史已保存到: {history_save_path}")
    except Exception as e:
        config['logger'].error(f"保存训练历史失败: {e}")
        raise





# 完整的使用流程
def main():
    """主函数 - 执行完整的模型训练和评估流程
    
    Returns:
        tuple: (trained_model, history)
    """
    try:
        # 1. 加载配置
        config = load_config()
        
        # 2. 设置日志
        logger = setup_logging(config)
        config['logger'] = logger  # 将logger添加到config字典中
        
        # 3. 设置设备
        device = setup_device(config)
        logger.info(f"使用设备: {device}")
        
        # 创建模型保存目录
        try:
            model_dir = Path(config["paths"]["model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"已创建模型保存目录: {model_dir}")
        except Exception as e:
            logger.error(f"创建模型保存目录失败: {e}")
            raise
        
        # 4. 加载数据
        train_df, val_df, test_df = load_data(config)
        
        # 5. 提取特征
        X_train, X_val, X_test, y_train, y_val, y_test, mean, std, y_mean, y_std = extract_features(config, [train_df, val_df, test_df])
        
        print(f"训练集大小: {X_train.shape[0]} 样本")
        print(f"验证集大小: {X_val.shape[0]} 样本")
        print(f"测试集大小: {X_test.shape[0]} 样本")
        print(f"特征维度: {X_train.shape[1]}")
        
        # 6. 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(config, X_train, X_val, X_test, y_train, y_val, y_test, mean, std, y_mean, y_std)
        
        # 7. 初始化模型
        model = initialize_model(config)
        
        # 8. 训练模型
        try:
            from train import train_transformer
            import torch.optim as optim
            
            # 初始化优化器
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=float(config['training']['weight_decay'])
            )
            
            # 初始化学习率调度器 - 自动预热余弦退火
            lr_scheduler_config = config["training"].get("lr_scheduler", {})
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                T_max=config["training"]["num_epochs"],  # 总epoch数作为退火周期
                min_lr=lr_scheduler_config.get("min_lr", 1e-6),  # 最小学习率
                warmup_lr_start=0.0  # 固定从0开始预热
            )
            
            if config['rank'] == 0:
                logger.info("开始训练Transformer...")
                logger.info(f"训练配置: 学习率={config['training']['learning_rate']}, 权重衰减={config['training']['weight_decay']}, 训练轮次={config['training']['num_epochs']}")
            
            trained_model, history = train_transformer(
                model, 
                train_loader, 
                val_loader, 
                optimizer, 
                scheduler, 
                device, 
                config, 
                logger, 
                rank=config['rank'],
                world_size=config['world_size']
            )
            
            if config['rank'] == 0:
                logger.info(f"训练完成！最佳验证损失: {min(history['val_loss']):.6f}")
                logger.info(f"最佳R²: {max(history['val_r2']):.4f}")
        except Exception as e:
            if config['rank'] == 0:
                logger.error(f"模型训练失败: {e}")
            raise
        
        print(f"训练完成！最佳验证损失: {min(history['val_loss']):.6f}")
        print(f"最佳R²: {max(history['val_r2']):.4f}")
        
        # 保存训练历史 - 只有主进程执行
        if config['rank'] == 0:
            save_training_history(config, history)
            logger.info("训练完成！")
        
        # 关闭分布式进程组
        if config.get('system', {}).get('distributed_training', False):
            import torch.distributed as dist
            dist.destroy_process_group()
        
        return trained_model, history
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise




if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="恒星质量预测模型")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    try:
        # 运行完整训练流程
        model, history = main()
    except Exception as e:
        print(f"全局异常捕获: {e}")
        traceback.print_exc()