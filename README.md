# 恒星质量预测插值软件 (M-Former)

本软件提供了一个简单易用的接口，用于使用训练好的Transformer模型（M-Former）预测恒星质量。该软件支持命令行和图形界面两种操作方式，适用于各种规模的恒星质量预测任务。

## 功能特点

- **双重操作模式**：命令行工具和图形界面，满足不同用户需求
- **灵活的输入方式**：支持单条参数预测、批量CSV文件处理和交互式输入
- **实时状态反馈**：显示模型加载状态和预测进度
- **用户友好界面**：直观的参数输入区域和清晰的结果展示
- **批量处理支持**：高效处理大量恒星数据

## 安装要求

- Python 3.8+
- PyTorch 2.0+

## 安装步骤

1. 克隆或下载项目代码
2. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 快速开始

### 命令行工具

#### 交互式模式

运行不带参数的命令进入交互式模式：

```bash
python interpolator.py
```

在交互式模式中：
- 输入各个参数值进行预测
- 输入'example'查看示例参数
- 输入'q'退出程序

#### 直接参数输入

```bash
python interpolator.py --logL 3.5 --logTeff 4.1 --FeH -0.2
```

#### 批量CSV文件处理

```bash
python interpolator.py --file input.csv --output predictions.csv
```

### 图形界面工具

启动图形界面：

```bash
python gui_interpolator.py
```

## 输入参数说明

| 参数名称 | 物理意义 | 单位 | 典型范围 |
|---------|---------|------|---------|
| logL    | 恒星光度的对数 | 太阳光度 | 2.0 - 4.5 |
| logTeff | 恒星有效温度的对数 | K | 3.5 - 4.5 |
| FeH     | 金属丰度 | [Fe/H] | -2.0 - 0.5 |

## 批量处理格式

### 输入CSV格式

```csv
logL,logTeff,FeH
3.5,4.1,-0.2
3.2,4.2,-0.1
4.0,3.9,0.0
```

### 输出CSV格式

输出文件将包含额外的`predicted_mass`列：

```csv
logL,logTeff,FeH,predicted_mass
3.5,4.1,-0.2,1.2345
3.2,4.2,-0.1,0.8765
4.0,3.9,0.0,2.1098
```

## 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| --logL | 恒星光度的对数 | None |
| --logTeff | 恒星有效温度的对数 | None |
| --FeH | 金属丰度 | None |
| --model | 模型文件路径 | models/best_model.pth |
| --device | 运行设备 | cpu |
| --file | 输入CSV文件路径 | None |
| --output | 输出CSV文件路径 | predictions.csv |

## 项目结构

```
m-former/
├── data/               # 数据集和标准化参数
├── evaluation_results/ # 评估结果和可视化
├── logs/               # 训练日志
├── models/             # 训练好的模型
├── monitoring/         # 监控和可视化数据
├── __pycache__/        # Python编译缓存
├── .env                # 环境变量
├── README.md           # 项目说明文档
├── README_wandb.md     # Weights & Biases使用说明
├── config.yaml         # 配置文件
├── gui_interpolator.py # 图形界面主文件
├── interpolator.py     # 命令行工具主文件
├── main.py             # 主入口文件
├── model.py            # 模型定义
├── monitoring.py       # 监控功能
└── requirements.txt    # 依赖列表
```

## 使用示例

### 单条预测示例

```
输入参数：
  logL: 3.5
  logTeff: 4.1
  FeH: -0.2

预测的恒星质量: 1.2345 太阳质量
```

### 批量处理示例

```
输入文件: input.csv
预测数量: 1000
结果文件: predictions.csv

预测质量范围:
  最小值: 0.5678 太阳质量
  最大值: 3.4567 太阳质量
  平均值: 1.1234 太阳质量
```

## 注意事项

1. 确保模型文件 `best_model.pth` 位于 `models` 目录下
2. 输入参数必须在合理的物理范围内，否则预测结果可能不准确
3. 批量处理时，输入文件必须包含所有必要的列（logL, logTeff, FeH）
4. 图形界面需要 Tkinter 库支持（已包含在Python标准库中）

## 技术支持

如果遇到问题，请检查：
1. 模型文件是否存在且完好
2. 依赖是否正确安装
3. 输入参数是否符合要求

## 许可证

MIT License
