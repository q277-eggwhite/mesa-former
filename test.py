#!/usr/bin/env python3
"""
恒星Transformer模型测试与可视化脚本
用于在测试集上评估已训练的模型性能并生成可视化报告
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from model import StellarTransformerWithPooling
from train import StellarDataset
# 从data模块导入标准化功能
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from data_standardization import load_standardization_params, denormalize_predictions
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set default font settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_config(config_path: str = "config.yaml"):
    """加载配置文件（支持JSON和YAML格式）"""
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        raise


def setup_logging(config):
    """设置日志记录"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("paths", {}).get("log_file", "test.log")
    
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("配置加载完成")
    return logger


def setup_device(config):
    """设置计算设备"""
    device_config = config.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and device_config == "auto" else device_config)
    return device


def load_test_data(config):
    """加载测试数据集"""
    data_config = config.get("data", {})
    paths_config = config.get("paths", {})
    
    data_dir = Path(paths_config.get("data_dir", "data"))
    test_file = data_config.get("test_file", "test_data.csv")
    sep = data_config.get("separator", "\\t")  # 默认使用制表符分隔
    
    logger = logging.getLogger(__name__)
    logger.info(f"从 {data_dir / test_file} 加载测试数据...")
    
    try:
        test_df = pd.read_csv(data_dir / test_file, sep=sep)
        logger.info(f"测试集大小: {len(test_df)}")
        
        feature_columns = data_config.get("feature_columns", ["log_L", "log10_age", "log_Teff", "[Fe/H]"])
        target_column = data_config.get("target_column", "log10_mass")  # 默认使用log10_mass作为目标列
        
        X_test = test_df[feature_columns].values
        y_test = test_df[target_column].values
        
        logger.info(f"特征维度: {X_test.shape[1]}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"测试数据加载失败: {e}")
        raise


def load_model(model_path, config, device):
    """加载训练好的模型"""
    logger = logging.getLogger(__name__)
    logger.info(f"从 {model_path} 加载模型...")
    
    try:
        model_config = config.get("model", {})
        model_params = model_config.get("params", {})
        
        model = StellarTransformerWithPooling(**model_params)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 从checkpoint中提取模型状态字典
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint
        
        # 处理DDP模型权重
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v
                
        # 使用strict=False忽略不匹配的参数
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        logger.info(f"模型加载成功，移动到 {device}")
        return model
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


def create_test_loader(X_test, y_test, config):
    """创建测试数据加载器"""
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("training", {}).get("num_workers", 0)
    
    # 将X_test和y_test组合成DataFrame
    feature_columns = config.get("data", {}).get("feature_columns", ["log_L", "log10_age", "log_Teff", "[Fe/H]"])
    target_column = config.get("data", {}).get("target_column", "log10_mass")
    
    test_df = pd.DataFrame(X_test, columns=feature_columns)
    test_df[target_column] = y_test
    
    # 加载标准化参数
    params = load_standardization_params(config)
    
    # 创建标准化参数张量
    mean = torch.tensor([
        params['log_L']['mean'],
        params['log_Teff']['mean'],
        params['[Fe/H]']['mean']
    ], dtype=torch.float32)
    
    std = torch.tensor([
        params['log_L']['std'],
        params['log_Teff']['std'],
        params['[Fe/H]']['std']
    ], dtype=torch.float32)
    
    y_mean = torch.tensor(params['log10_mass']['mean'], dtype=torch.float32)
    y_std = torch.tensor(params['log10_mass']['std'], dtype=torch.float32)
    
    # 创建测试数据集，启用标准化
    test_dataset = StellarDataset(
        test_df, 
        normalize=True, 
        mean=mean,
        std=std,
        y_mean=y_mean,
        y_std=y_std
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    logger = logging.getLogger(__name__)
    logger.info(f"测试数据加载器创建完成，批次大小: {batch_size}")
    return test_loader


def evaluate_model(model, test_loader, device, config=None, denormalize=False):
    """在测试集上评估模型性能"""
    logger = logging.getLogger(__name__)
    logger.info("开始在测试集上评估模型...")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # 解包特征：log_L, log_Teff, [Fe/H]
            logL = batch_features[:, 0]
            logTeff = batch_features[:, 1]
            FeH = batch_features[:, 2]
            
            pred_mass = model(logL, logTeff, FeH)
            
            predictions.append(pred_mass.cpu())
            targets.append(batch_targets.cpu())
    
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    
    # 初始化反标准化后的预测值和目标值
    predictions_denorm = None
    targets_denorm = None
    
    # 反标准化预测值和目标值
    if denormalize and config is not None:
        # 加载标准化参数
        params = load_standardization_params(config)
        y_mean = params['log10_mass']['mean']
        y_std = params['log10_mass']['std']
        
        # 反标准化
        predictions_denorm = denormalize_predictions(predictions, y_mean, y_std)
        targets_denorm = denormalize_predictions(targets, y_mean, y_std)
        
        # 计算评估指标（使用反标准化后的值）
        mse = mean_squared_error(targets_denorm, predictions_denorm)
        mae = mean_absolute_error(targets_denorm, predictions_denorm)
        r2 = r2_score(targets_denorm, predictions_denorm)
        relative_errors = np.abs((predictions_denorm - targets_denorm) / targets_denorm)
        residuals = targets_denorm - predictions_denorm
    else:
        # 计算评估指标（使用标准化后的值）
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        relative_errors = np.abs((predictions - targets) / targets)
        residuals = targets - predictions
    
    results = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2,
        'mean_relative_error': np.mean(relative_errors) * 100,
        'median_relative_error': np.median(relative_errors) * 100,
        'residual_mean': np.mean(residuals),
        'residual_std': np.std(residuals),
        'predictions': predictions,
        'targets': targets,
        'predictions_denorm': predictions_denorm,
        'targets_denorm': targets_denorm
    }
    
    logger.info(f"测试集评估结果: MSE={mse:.6f}, RMSE={results['rmse']:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
    return results


def create_visualizations(targets, predictions, X_test=None, save_dir=None, show_plots=True):
    """创建所有可视化图表"""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 预测值vs真实值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    r2 = r2_score(targets, predictions)
    plt.xlabel('log True Mass (log M☉)', fontsize=12)
    plt.ylabel('log Predicted Mass (log M☉)', fontsize=12)
    plt.title(f'log True vs Predicted Values (R² = {r2:.4f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(save_dir / "predictions.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. 残差图和误差分布
    residuals = targets - predictions
    relative_errors = (predictions - targets) / targets * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs True Values
    axes[0, 0].scatter(targets, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('True Mass (M☉)')
    axes[0, 0].set_ylabel('Residuals (True - Predicted)')
    axes[0, 0].set_title('Residuals vs True Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual Distribution
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals (True - Predicted)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relative Error Distribution
    axes[1, 0].hist(relative_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Relative Error (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Relative Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Relative Error Box Plot
    axes[1, 1].boxplot(relative_errors, vert=False)
    axes[1, 1].set_xlabel('Relative Error (%)')
    axes[1, 1].set_title('Relative Error Box Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / "residuals_and_errors.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 3. 特征相关性分析（如果提供了测试数据）
    if X_test is not None:
        feature_names = ["log_L", "log_Teff", "[Fe/H]"]
        
        plt.figure(figsize=(15, 8))
        for i, feature_name in enumerate(feature_names):
            plt.subplot(1, 3, i+1)
            plt.scatter(X_test[:, i], np.abs(relative_errors), alpha=0.6)
            plt.xlabel(feature_name, fontsize=12)
            plt.ylabel('Relative Error', fontsize=12)
            plt.title(f'{feature_name} vs Relative Error', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / "feature_correlation.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def plot_training_history(history: Dict[str, List[float]], save_path=None, show=True):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² Score
    if 'val_r2' in history:
        axes[0, 1].plot(history['val_r2'], label='Validation R²')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_title('Validation R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history:
        axes[1, 0].plot(history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Change')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Training and Validation R² Score
    if 'train_r2' in history:
        axes[1, 1].plot(history['train_r2'], label='Training R²')
        if 'val_r2' in history:
            axes[1, 1].plot(history['val_r2'], label='Validation R²')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('Training and Validation R² Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_models(model_results: Dict[str, Dict], save_dir=None, show=True):
    """比较多个模型的性能"""
    # 提取指标
    metrics_df = pd.DataFrame({
        name: results['metrics'] 
        for name, results in model_results.items()
    }).T
    
    print("模型性能比较:")
    print(metrics_df.round(6))
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(save_dir / "model_comparison.csv")
    
    # 绘制指标比较图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_df['mse'].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('MSE Comparison')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    metrics_df['rmse'].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    metrics_df['mae'].plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('MAE Comparison')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3)
    
    metrics_df['r2'].plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('R² Comparison')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # 绘制所有模型的预测散点图
    plt.figure(figsize=(15, 5))
    
    for i, (name, results) in enumerate(model_results.items()):
        plt.subplot(1, len(model_results), i+1)
        y_true = results['targets']
        y_pred = results['predictions']
        
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('True Mass (M☉)')
        plt.ylabel('Predicted Mass (M☉)')
        plt.title(f'{name}\n(R² = {r2:.4f})')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / "all_models_predictions.png", dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_report(model_name: str, metrics: Dict[str, float], 
                   y_true: np.ndarray, y_pred: np.ndarray, save_path=None):
    """生成模型评估报告"""
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    report = f"""
# 模型评估报告: {model_name}

## 评估指标

| 指标 | 值 |
|------|-----|
| 均方误差 (MSE) | {metrics['mse']:.6f} |
| 均方根误差 (RMSE) | {metrics['rmse']:.6f} |
| 平均绝对误差 (MAE) | {metrics['mae']:.6f} |
| R² 分数 | {metrics['r2']:.4f} |
| 平均绝对百分比误差 (MAPE) | {mape:.2f}% |
| 残差均值 | {metrics['residual_mean']:.6f} |
| 残差标准差 | {metrics['residual_std']:.6f} |

## 数据集信息

- 样本数量: {len(y_true)}
- 真实值范围: [{y_true.min():.4f}, {y_true.max():.4f}]
- 预测值范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]

## 分析与结论

1. **模型性能**:
   - R² 分数为 {metrics['r2']:.4f}，表明模型解释了目标变量 {metrics['r2']*100:.2f}% 的变异。
   - 平均绝对百分比误差为 {mape:.2f}%，表示平均预测误差程度。
   
2. **残差分析**:
   - 残差均值为 {metrics['residual_mean']:.6f}，接近0，表明模型没有明显的系统性偏差。
   - 残差标准差为 {metrics['residual_std']:.6f}，表示预测的离散程度。

## 可视化图表

- 预测值 vs 真实值散点图
- 残差图
- 误差分布图
- 特征相关性分析
    """
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report


def save_results(results, save_path):
    """保存评估结果到JSON文件"""
    logger = logging.getLogger(__name__)
    
    serializable_results = {}
    for key, value in results.items():
        if key in ['predictions', 'targets', 'predictions_denorm', 'targets_denorm']:
            if value is not None:
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = None
        else:
            if value is not None:
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = None
    
    try:
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"评估结果已保存到: {save_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        raise


def save_predictions_csv(results, save_path):
    """保存预测结果到CSV文件"""
    logger = logging.getLogger(__name__)
    
    try:
        # 使用反标准化后的值（如果存在），否则使用原始值
        if results['targets_denorm'] is not None and results['predictions_denorm'] is not None:
            true_log_mass = results['targets_denorm']
            pred_log_mass = results['predictions_denorm']
        else:
            true_log_mass = results['targets']
            pred_log_mass = results['predictions']
        
        # 将对数质量转换为原始质量（10^log10_mass）
        true_mass = 10 ** true_log_mass
        pred_mass = 10 ** pred_log_mass
        
        df = pd.DataFrame({
            'true_mass': true_mass,
            'pred_mass': pred_mass,
            'absolute_error': np.abs(pred_mass - true_mass),
            'relative_error': np.abs((pred_mass - true_mass) / true_mass) * 100
        })
        df.to_csv(save_path, index=False)
        logger.info(f"预测结果已保存到: {save_path}")
    except Exception as e:
        logger.error(f"保存预测结果失败: {e}")
        raise


def evaluate_single_model(model, test_loader, device, model_name="model", save_dir="./evaluation_results", show_plots=True):
    """评估单个模型的便捷函数"""
    logger = logging.getLogger(__name__)
    logger.info(f"评估模型: {model_name}")
    
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    
    results = {
        'metrics': metrics,
        'predictions': predictions,
        'targets': targets
    }
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    create_visualizations(targets, predictions, show_plots=show_plots, save_dir=plots_dir)
    report = generate_report(model_name, metrics, targets, predictions, save_path=save_dir / f"{model_name}_evaluation_report.md")
    save_results(results, save_dir / f"{model_name}_results.json")
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'targets': targets,
        'report': report
    }


def evaluate_multiple_models(model_dict: Dict[str, Any], test_loader, device, save_dir="./evaluation_results", show_plots=True):
    """评估多个模型的便捷函数"""
    logger = logging.getLogger(__name__)
    logger.info(f"评估 {len(model_dict)} 个模型...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_results = {}
    
    for name, model in model_dict.items():
        metrics, predictions, targets = evaluate_model(model, test_loader, device)
        
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
        
        model_results[name] = results
    
    compare_models(model_results, save_dir=save_dir, show=show_plots)
    
    for name, results in model_results.items():
        report = generate_report(name, results['metrics'], results['targets'], results['predictions'], 
                               save_path=save_dir / f"{name}_evaluation_report.md")
        results['report'] = report
        save_results(results, save_dir / f"{name}_results.json")
    
    return model_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒星Transformer模型测试与可视化脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='模型文件路径')
    parser.add_argument('--output', type=str, default='evaluation_results', help='结果保存目录')
    parser.add_argument('--visualize', action='store_false', default=True, help='不生成可视化报告（默认生成）')
    parser.add_argument('--show-plots', action='store_true', help='显示可视化图表')
    parser.add_argument('--history', type=str, default=None, help='训练历史文件路径')
    parser.add_argument('--skip-evaluation', action='store_true', help='跳过模型评估步骤，仅生成可视化报告')
    
    args = parser.parse_args()
    
    try:
        # 1. 加载配置
        config = load_config(args.config)
        
        # 2. 设置日志
        logger = setup_logging(config)
        
        # 3. 设置设备
        device = setup_device(config)
        logger.info(f"使用设备: {device}")
        
        # 4. 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. 加载测试数据
        X_test, y_test = load_test_data(config)
        
        # 6. 检查是否跳过评估
        if args.skip_evaluation:
            # 尝试从已有结果文件加载评估结果
            results_file = output_dir / "test_results.json"
            if results_file.exists():
                logger.info(f"跳过评估，从 {results_file} 加载已有结果")
                with open(results_file, 'r') as f:
                    results = json.load(f)
                # 将列表转换回numpy数组
                for key in ['predictions', 'targets', 'predictions_denorm', 'targets_denorm']:
                    if results[key] is not None:
                        results[key] = np.array(results[key])
            else:
                logger.error(f"跳过评估失败：结果文件 {results_file} 不存在")
                raise FileNotFoundError(f"结果文件 {results_file} 不存在，无法跳过评估")
        else:
            # 正常评估流程
            # 6. 创建测试数据加载器
            test_loader = create_test_loader(X_test, y_test, config)
            
            # 7. 加载模型
            model = load_model(args.model, config, device)
            
            # 8. 评估模型，启用反标准化
            results = evaluate_model(model, test_loader, device, config=config, denormalize=True)
            
            # 9. 保存结果
            results_file = output_dir / "test_results.json"
            save_results(results, results_file)
            
            # 10. 保存预测结果到CSV
            predictions_file = output_dir / "predictions.csv"
            save_predictions_csv(results, predictions_file)
        
        # 11. 生成可视化报告
        if args.visualize or args.show_plots:
            plots_dir = output_dir / "plots"
            # 使用反标准化后的值进行可视化
            viz_targets = results['targets_denorm'] if results['targets_denorm'] is not None else results['targets']
            viz_predictions = results['predictions_denorm'] if results['predictions_denorm'] is not None else results['predictions']
            create_visualizations(viz_targets, viz_predictions, X_test, 
                                save_dir=plots_dir, show_plots=args.show_plots)
        
        # 12. 绘制训练历史（如果提供）
        if args.history:
            try:
                with open(args.history, 'r') as f:
                    history = json.load(f)
                plot_training_history(history, save_path=output_dir / "plots" / "training_history.png", show=args.show_plots)
            except Exception as e:
                logger.warning(f"无法加载或绘制训练历史: {e}")
        
        # 13. 生成评估报告
        report = generate_report("StellarTransformer", results, results['targets'], results['predictions'], 
                               save_path=output_dir / "evaluation_report.md")
        
        # 14. 打印摘要
        print("\n" + "="*50)
        print("模型测试结果摘要")
        print("="*50)
        print(f"MSE: {results['mse']:.6f}")
        print(f"RMSE: {results['rmse']:.6f}")
        print(f"MAE: {results['mae']:.6f}")
        print(f"R²: {results['r2']:.4f}")
        print(f"平均相对误差: {results['mean_relative_error']:.2f}%")
        print(f"中位数相对误差: {results['median_relative_error']:.2f}%")
        print("="*50)
        
        logger.info("模型测试完成！")
        return results
    except Exception as e:
        logger.error(f"测试过程失败: {e}")
        raise


if __name__ == "__main__":
    results = main()