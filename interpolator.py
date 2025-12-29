#!/usr/bin/env python3
"""
恒星质量预测插值软件

功能：使用训练好的Transformer模型预测恒星质量
输入：恒星光度(logL)、有效温度(logTeff)、金属丰度(FeH)
输出：预测的恒星质量

使用方法：
1. 交互式输入：python interpolator.py
2. 命令行参数：python interpolator.py --log_L 3.5 --log_Teff 4.1 --Fe_H -0.2
3. 从文件读取：python interpolator.py --file input.csv
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path

# 导入模型
from model import StellarTransformerWithPooling

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_standardization_params(params_path: str = 'data/standardization_params.json') -> Dict[str, Dict[str, float]]:
    """加载标准化参数
    
    Args:
        params_path: 标准化参数文件路径
        
    Returns:
        标准化参数字典
    """
    try:
        logger.info(f"从 {params_path} 加载标准化参数...")
        with open(params_path, 'r') as f:
            params = json.load(f)
        logger.info("标准化参数加载成功")
        return params
    except Exception as e:
        logger.warning(f"加载标准化参数失败: {e}")
        logger.warning("将使用默认值，预测结果可能不准确")
        # 返回默认参数
        return {
            'log_L': {'mean': 0.0, 'std': 1.0},
            'log_Teff': {'mean': 0.0, 'std': 1.0},
            '[Fe/H]': {'mean': 0.0, 'std': 1.0},
            'log10_mass': {'mean': 0.0, 'std': 1.0}
        }

def load_model(model_path: str = 'models/best_model.pth', device: str = 'cpu') -> torch.nn.Module:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 运行设备
        
    Returns:
        加载好的模型
    """
    try:
        logger.info(f"从 {model_path} 加载模型...")
        
        # 创建模型实例（使用默认参数）
        model = StellarTransformerWithPooling()
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 获取模型状态字典（实际权重存储在'model_state_dict'键中）
        if 'model_state_dict' in checkpoint:
            model_weights = checkpoint['model_state_dict']
            logger.info("从 'model_state_dict' 键加载模型权重")
        else:
            model_weights = checkpoint
            logger.info("从顶层加载模型权重")
        
        # 处理DDP模型权重
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_weights.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 移除'module.'前缀
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        
        logger.info(f"模型加载成功，运行在 {device} 上")
        return model
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def predict_mass(model: torch.nn.Module, log_L: float, log_Teff: float, Fe_H: float, 
                device: str = 'cpu', standardization_params: Optional[Dict] = None, return_log: bool = False) -> float:
    """使用模型预测恒星质量
    
    Args:
        model: 加载好的模型
        log_L: 恒星光度的对数 (log_L)
        log_Teff: 恒星有效温度的对数 (log_Teff)
        Fe_H: 金属丰度 ([Fe/H])
        device: 运行设备
        standardization_params: 标准化参数字典
        return_log: 是否返回对数质量，否则返回线性质量
        
    Returns:
        预测的恒星质量（对数质量或线性质量）
    """
    # 加载标准化参数
    if standardization_params is None:
        standardization_params = load_standardization_params()
    
    # 对所有三个输入特征进行标准化，与训练逻辑一致
    feature_means = np.array([
        standardization_params['log_L']['mean'],
        standardization_params['log_Teff']['mean'],
        standardization_params['[Fe/H]']['mean']
    ])
    feature_stds = np.array([
        standardization_params['log_L']['std'],
        standardization_params['log_Teff']['std'],
        standardization_params['[Fe/H]']['std']
    ])
    
    # 标准化所有三个特征
    features_std = (np.array([log_L, log_Teff, Fe_H]) - feature_means) / feature_stds
    
    # 将输入转换为张量
    logL_tensor = torch.tensor([features_std[0]], dtype=torch.float32, device=device)
    logTeff_tensor = torch.tensor([features_std[1]], dtype=torch.float32, device=device)
    FeH_tensor = torch.tensor([features_std[2]], dtype=torch.float32, device=device)
    
    # 进行预测
    with torch.no_grad():
        prediction = model(logL_tensor, logTeff_tensor, FeH_tensor)
    
    # 对预测结果进行反标准化
    y_mean = standardization_params['log10_mass']['mean']
    y_std = standardization_params['log10_mass']['std']
    prediction_denorm = prediction.item() * y_std + y_mean
    
    if return_log:
        return prediction_denorm
    else:
        # 转换为线性质量
        return 10 ** prediction_denorm

def predict_from_dataframe(model: torch.nn.Module, df: pd.DataFrame, device: str = 'cpu') -> pd.DataFrame:
    """从DataFrame批量预测恒星质量
    
    Args:
        model: 加载好的模型
        df: 包含输入特征的DataFrame，支持以下列名格式：
            - 格式1: ['logL', 'logTeff', 'FeH', 'log10_age']
            - 格式2: ['log_L', 'log10_age', 'log_Teff', '[Fe/H]']
        device: 运行设备
        
    Returns:
        添加了预测质量列的DataFrame
    """
    # 加载标准化参数
    standardization_params = load_standardization_params()
    
    # 列名映射（支持两种格式）
    column_mapping = {
        'log_L': 'logL',
        'log_Teff': 'logTeff',
        '[Fe/H]': 'FeH'
    }
    
    # 创建一个副本，避免修改原始数据
    df_processed = df.copy()
    
    # 重命名列以匹配模型需要的格式
    df_processed = df_processed.rename(columns=column_mapping)
    
    # 检查必要的列
    required_columns = ['logL', 'logTeff', 'FeH']
    for col in required_columns:
        if col not in df_processed.columns:
            raise ValueError(f"DataFrame缺少必要的列: {col}。支持的格式：['logL', 'logTeff', 'FeH'] 或 ['log_L', 'log_Teff', '[Fe/H]']")
    
    # 提取特征
    logL_values = df_processed['logL'].values
    logTeff_values = df_processed['logTeff'].values
    FeH_values = df_processed['FeH'].values
    
    # 对所有三个输入特征进行标准化，与训练逻辑一致
    feature_means = np.array([
        standardization_params['log_L']['mean'],
        standardization_params['log_Teff']['mean'],
        standardization_params['[Fe/H]']['mean']
    ])
    feature_stds = np.array([
        standardization_params['log_L']['std'],
        standardization_params['log_Teff']['std'],
        standardization_params['[Fe/H]']['std']
    ])
    
    # 标准化所有三个特征
    features = np.array([logL_values, logTeff_values, FeH_values]).T
    features_std = (features - feature_means) / feature_stds
    
    # 批量预测
    logL_tensor = torch.tensor(features_std[:, 0], dtype=torch.float32, device=device)
    logTeff_tensor = torch.tensor(features_std[:, 1], dtype=torch.float32, device=device)
    FeH_tensor = torch.tensor(features_std[:, 2], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        predictions = model(logL_tensor, logTeff_tensor, FeH_tensor)
    
    # 对预测结果进行反标准化
    y_mean = standardization_params['log10_mass']['mean']
    y_std = standardization_params['log10_mass']['std']
    predictions_denorm = predictions.cpu().numpy() * y_std + y_mean
    
    # 转换为线性质量
    predicted_mass = 10 ** predictions_denorm
    
    # 将结果添加到DataFrame
    df_with_pred = df.copy()
    df_with_pred['predicted_mass'] = predicted_mass
    df_with_pred['predicted_log_mass'] = predictions_denorm  # 可选：添加对数质量列
    
    return df_with_pred

def interactive_mode(model: torch.nn.Module, device: str = 'cpu'):
    """交互式输入模式
    
    Args:
        model: 加载好的模型
        device: 运行设备
    """
    # 加载标准化参数
    standardization_params = load_standardization_params()
    
    print("\n=== 恒星质量预测插值软件 ===")
    print("请输入恒星的物理参数 (符合训练数据格式):")
    print("（输入'q'退出，输入'example'查看示例）")
    
    while True:
        try:
            # 获取输入
            log_L_input = input("\nlog_L (恒星光度的对数): ")
            if log_L_input.lower() == 'q':
                break
            elif log_L_input.lower() == 'example':
                print("示例：")
                print("log_L = 3.5, log10_age = 4.0, log_Teff = 4.1, [Fe/H] = -0.2")
                print("对应的预测质量约为：1.2 太阳质量")
                continue
            
            log_Teff_input = input("log_Teff (恒星有效温度的对数): ")
            if log_Teff_input.lower() == 'q':
                break
                
            Fe_H_input = input("[Fe/H] (金属丰度): ")
            if Fe_H_input.lower() == 'q':
                break
            
            # 转换为数值
            log_L = float(log_L_input)
            log_Teff = float(log_Teff_input)
            Fe_H = float(Fe_H_input)
            
            # 进行预测
            predicted_mass = predict_mass(model, log_L, log_Teff, Fe_H, device, standardization_params)
            
            # 显示结果
            print(f"\n=== 预测结果 ===")
            print(f"输入参数：")
            print(f"  log_L: {log_L:.2f}")
            print(f"  log_Teff: {log_Teff:.2f}")
            print(f"  [Fe/H]: {Fe_H:.2f}")
            print(f"\n预测的恒星质量: {predicted_mass:.4f} 太阳质量")
            
        except ValueError as e:
            print(f"输入错误: {e}")
            print("请输入有效的数值")
        except Exception as e:
            print(f"发生错误: {e}")
            print("请重试")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='恒星质量预测插值软件')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='模型文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备 (cpu/gpu/cuda)')
    parser.add_argument('--standardization-params', type=str, default='data/standardization_params.json', help='标准化参数文件路径')
    
    # 命令行输入模式
    parser.add_argument('--log_L', type=float, help='恒星光度的对数')
    parser.add_argument('--log_Teff', type=float, help='恒星有效温度的对数')
    parser.add_argument('--Fe_H', type=float, help='金属丰度')
    
    # 文件输入模式
    parser.add_argument('--file', type=str, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='predictions.csv', help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    # 加载标准化参数
    standardization_params = load_standardization_params(args.standardization_params)
    
    # 加载模型
    model = load_model(args.model, args.device)
    
    # 处理不同的输入模式
    if args.log_L is not None and args.log_Teff is not None and args.Fe_H is not None:
        # 命令行参数输入模式
        predicted_mass = predict_mass(model, args.log_L, args.log_Teff, args.Fe_H, args.device, standardization_params)
        print(f"\n=== 预测结果 ===")
        print(f"输入参数：")
        print(f"  log_L: {args.log_L:.2f}")
        print(f"  log_Teff: {args.log_Teff:.2f}")
        print(f"  [Fe/H]: {args.Fe_H:.2f}")
        print(f"\n预测的恒星质量: {predicted_mass:.4f} 太阳质量")
        
    elif args.file:
        # 文件输入模式
        if not os.path.exists(args.file):
            print(f"错误: 文件 {args.file} 不存在")
            return
        
        try:
            # 读取输入文件
            df = pd.read_csv(args.file)
            logger.info(f"从 {args.file} 读取了 {len(df)} 条数据")
            
            # 进行预测
            df_with_pred = predict_from_dataframe(model, df, args.device)
            
            # 保存结果
            df_with_pred.to_csv(args.output, index=False)
            logger.info(f"预测结果已保存到 {args.output}")
            
            # 显示摘要
            print(f"\n=== 预测摘要 ===")
            print(f"输入文件: {args.file}")
            print(f"预测数量: {len(df_with_pred)}")
            print(f"结果文件: {args.output}")
            print(f"\n预测质量范围:")
            print(f"  最小值: {df_with_pred['predicted_mass'].min():.4f} 太阳质量")
            print(f"  最大值: {df_with_pred['predicted_mass'].max():.4f} 太阳质量")
            print(f"  平均值: {df_with_pred['predicted_mass'].mean():.4f} 太阳质量")
            
        except Exception as e:
            logger.error(f"处理文件时出错: {e}")
            print(f"错误: {e}")
            return
            
    else:
        # 交互式输入模式
        interactive_mode(model, args.device)

if __name__ == "__main__":
    main()