import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
from pathlib import Path
import logging


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =========================================
# 数据预处理功能
# =========================================

def preprocess_data(data_path, output_dir="."):
    """
    从原始数据生成训练/验证/测试集并计算标准化参数
    
    Args:
        data_path: 原始数据文件路径
        output_dir: 输出目录路径
    
    Returns:
        tuple: (train_df, val_df, test_df, standardization_params)
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确保数据路径是相对于脚本目录的
    if not os.path.isabs(data_path):
        data_path = os.path.join(script_dir, data_path)
    
    # 确保输出目录是相对于脚本目录的
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    logger.info(f"从 {data_path} 读取原始数据...")
    df = pd.read_csv(data_path, sep="\t")
    
    logger.info("原始数据信息：")
    logger.info(f"数据形状：{df.shape}")
    logger.info(f"列名：{list(df.columns)}")
    
    # 步骤1：特征转换
    logger.info("\n=== 特征转换 ===")
    
    # 年龄取对数
    df['log10_age'] = np.log10(df['star_age'])
    
    # 质量取对数作为目标变量
    df['log10_mass'] = np.log10(df['star_mass'])
    
    # 步骤2：构建完整数据集（包含原始特征和转换后的特征）
    logger.info("\n=== 构建完整数据集 ===")
    
    # 定义最终数据集的列名（包含原始特征和转换后的特征）
    final_columns = ['log_L', 'log10_age', 'log_Teff', '[Fe/H]', 'log10_mass', 'star_age', 'star_mass']
    
    # 步骤3：数据集划分
    logger.info("\n=== 数据集划分 ===")
    
    # 划分训练集(80%)、验证集(10%)和测试集(10%)
    train_df, temp_df = train_test_split(df[final_columns], test_size=0.2, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)
    
    logger.info(f"训练集大小：{len(train_df)} 样本")
    logger.info(f"验证集大小：{len(val_df)} 样本")
    logger.info(f"测试集大小：{len(test_df)} 样本")
    
    # 步骤4：物理一致性检查
    logger.info("\n=== 物理一致性检查 ===")
    
    # 检查质光关系：log_L和log10_mass的相关性
    correlation = np.corrcoef(train_df['log_L'], train_df['log10_mass'])[0, 1]
    logger.info(f"质光关系（log_L vs log10_mass）相关性：{correlation:.4f}")
    
    # 检查金属丰度与质量的关系
    correlation_metallicity_mass = np.corrcoef(train_df['[Fe/H]'], train_df['log10_mass'])[0, 1]
    logger.info(f"金属丰度与质量相关性：{correlation_metallicity_mass:.4f}")
    
    # 步骤5：计算标准化参数
    logger.info("\n=== 计算标准化参数 ===")
    
    # 定义特征列和目标列
    feature_columns = ['log_L', 'log_Teff', '[Fe/H]']
    target_column = 'log10_mass'
    
    # 计算特征标准化参数
    train_features = train_df[feature_columns].values
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    
    # 计算目标变量标准化参数
    train_targets = train_df[target_column].values
    target_mean = train_targets.mean()
    target_std = train_targets.std()
    
    # 构建标准化参数字典
    standardization_params = {
        'log_L': {'mean': float(feature_mean[0]), 'std': float(feature_std[0])},
        'log_Teff': {'mean': float(feature_mean[1]), 'std': float(feature_std[1])},
        '[Fe/H]': {'mean': float(feature_mean[2]), 'std': float(feature_std[2])},
        'log10_mass': {'mean': float(target_mean), 'std': float(target_std)}
    }
    
    logger.info(f"特征均值：{feature_mean}")
    logger.info(f"特征标准差：{feature_std}")
    logger.info(f"目标均值：{target_mean}")
    logger.info(f"目标标准差：{target_std}")
    
    # 步骤6：保存原始转换后的数据集
    logger.info("\n=== 保存数据集 ===")
    
    # 保存为CSV格式
    # 简化路径，移除多余的./
    train_path = os.path.normpath(os.path.join(output_dir, 'train_data.csv'))
    val_path = os.path.normpath(os.path.join(output_dir, 'val_data.csv'))
    test_path = os.path.normpath(os.path.join(output_dir, 'test_data.csv'))
    
    train_df.to_csv(train_path, index=False, sep='\t')
    val_df.to_csv(val_path, index=False, sep='\t')
    test_df.to_csv(test_path, index=False, sep='\t')
    
    logger.info(f"训练集已保存到：{train_path}")
    logger.info(f"验证集已保存到：{val_path}")
    logger.info(f"测试集已保存到：{test_path}")
    
    # 步骤7：保存标准化参数
    params_path = os.path.normpath(os.path.join(output_dir, 'standardization_params.json'))
    with open(params_path, 'w') as f:
        json.dump(standardization_params, f, indent=4)
    
    logger.info(f"标准化参数已保存到：{params_path}")
    
    logger.info("\n=== 数据预处理完成 ===")
    logger.info("所有步骤已完成，数据准备就绪可以输入Transformer模型！")
    
    return train_df, val_df, test_df, standardization_params


# =========================================
# 标准化参数管理功能
# =========================================

def load_standardization_params(config):
    """加载标准化参数"""
    paths_config = config.get("paths", {})
    data_dir = Path(paths_config.get("data_dir", "data"))
    params_file = paths_config.get("standardization_params", "standardization_params.json")
    
    try:
        # 首先尝试从文件加载标准化参数
        if (data_dir / params_file).exists():
            logger.info(f"从 {data_dir / params_file} 加载标准化参数...")
            with open(data_dir / params_file, 'r') as f:
                params = json.load(f)
            logger.info("标准化参数加载成功")
            return params
        else:
            # 如果文件不存在，从数据中重新计算
            logger.info("标准化参数文件不存在，从训练数据中计算...")
            train_file = paths_config.get("train_file", "train_data.csv")
            train_df = pd.read_csv(data_dir / train_file, sep='\t')
            
            feature_columns = config.get("data", {}).get("feature_columns", ["log_L", "log_Teff", "[Fe/H]"])
            target_column = config.get("data", {}).get("target_column", "log10_mass")
            
            # 计算特征标准化参数
            features = train_df[feature_columns].values
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            
            # 计算目标变量标准化参数
            targets = train_df[target_column].values
            y_mean = targets.mean()
            y_std = targets.std()
            
            params = {
                'log_L': {'mean': float(mean[0]), 'std': float(std[0])},
                'log10_age': {'mean': float(mean[1]), 'std': float(std[1])},
                'log_Teff': {'mean': float(mean[2]), 'std': float(std[2])},
                '[Fe/H]': {'mean': float(mean[3]), 'std': float(std[3])},
                'log10_mass': {'mean': float(y_mean), 'std': float(y_std)}
            }
            
            # 保存计算得到的标准化参数
            with open(data_dir / params_file, 'w') as f:
                json.dump(params, f, indent=4)
            logger.info(f"标准化参数已计算并保存到 {data_dir / params_file}")
            return params
    except Exception as e:
        logger.error(f"加载或计算标准化参数失败: {e}")
        raise


def calculate_normalization_params(train_df, feature_columns, target_column):
    """计算标准化参数"""
    # 计算特征标准化参数
    features = train_df[feature_columns].values
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    
    # 计算目标变量标准化参数
    targets = train_df[target_column].values
    y_mean = targets.mean()
    y_std = targets.std()
    
    return {
        'feature_mean': mean,
        'feature_std': std,
        'target_mean': y_mean,
        'target_std': y_std
    }


def save_normalization_params(params, config):
    """保存标准化参数"""
    paths_config = config.get("paths", {})
    data_dir = Path(paths_config.get("data_dir", "data"))
    params_file = paths_config.get("standardization_params", "standardization_params.json")
    
    # 将numpy数组转换为Python原生类型
    serializable_params = {
        'log_L': {'mean': float(params['feature_mean'][0]), 'std': float(params['feature_std'][0])},
        'log10_age': {'mean': float(params['feature_mean'][1]), 'std': float(params['feature_std'][1])},
        'log_Teff': {'mean': float(params['feature_mean'][2]), 'std': float(params['feature_std'][2])},
        '[Fe/H]': {'mean': float(params['feature_mean'][3]), 'std': float(params['feature_std'][3])},
        'log10_mass': {'mean': float(params['target_mean']), 'std': float(params['target_std'])}
    }
    
    with open(data_dir / params_file, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    
    logger.info(f"标准化参数已保存到 {data_dir / params_file}")


# =========================================
# 标准化操作功能
# =========================================

def normalize_features(features, mean, std):
    """对特征进行标准化"""
    return (features - mean) / std


def normalize_target(target, y_mean, y_std):
    """对目标变量进行标准化"""
    return (target - y_mean) / y_std


def denormalize_predictions(predictions, y_mean, y_std):
    """反标准化预测值"""
    return (predictions * y_std) + y_mean


# =========================================
# 主函数
# =========================================
if __name__ == "__main__":
    # 示例用法 - 使用相对路径
    # 无论从哪个目录运行，都使用相对于data目录的相对路径
    data_path = "training_data.txt"  # 相对于data目录的相对路径
    output_dir = "."                 # 输出到当前目录（即data目录）
    
    train_df, val_df, test_df, params = preprocess_data(data_path, output_dir=output_dir)
    
    # 打印标准化参数
    logger.info("\n标准化参数：")
    logger.info(json.dumps(params, indent=2))
    
    # 示例：使用标准化功能
    sample_feature = np.array([0.9432, 8.8043, 3.9674, -1.0])  # log_L, log10_age, log_Teff, [Fe/H]
    sample_target = np.array([0.1522])  # log10_mass
    
    # 计算标准化参数
    feature_means = np.array([params['log_L']['mean'], params['log10_age']['mean'], 
                             params['log_Teff']['mean'], params['[Fe/H]']['mean']])
    feature_stds = np.array([params['log_L']['std'], params['log10_age']['std'], 
                            params['log_Teff']['std'], params['[Fe/H]']['std']])
    y_mean = params['log10_mass']['mean']
    y_std = params['log10_mass']['std']
    
    # 应用标准化
    norm_feature = normalize_features(sample_feature, feature_means, feature_stds)
    norm_target = normalize_target(sample_target, y_mean, y_std)
    
    logger.info("\n示例标准化应用：")
    logger.info(f"原始特征：{sample_feature}")
    logger.info(f"标准化特征：{norm_feature}")
    logger.info(f"原始目标：{sample_target}")
    logger.info(f"标准化目标：{norm_target}")
    
    # 反标准化示例
    denorm_prediction = denormalize_predictions(norm_target, y_mean, y_std)
    logger.info(f"反标准化预测值：{denorm_prediction}")
    logger.info(f"与原始目标的差异：{denorm_prediction - sample_target}")
