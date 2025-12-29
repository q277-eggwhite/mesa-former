# 恒星质量预测模型 - WandB实时监控

本项目实现了基于Transformer架构的恒星质量预测模型，并集成了WandB实时性能监控功能，提供全面的训练过程可视化与监控。

## 功能特点

- 🌟 基于Transformer的恒星质量预测模型
- 📊 WandB实时性能监控仪表板
- 🔍 系统资源监控（CPU、内存、GPU使用率）
- 📈 训练指标实时可视化（损失、R²分数、学习率等）
- 🚨 异常情况报警（损失增加、训练停滞等）
- 📝 结构化日志记录
- 🧪 模型测试与评估
- 📊 预测结果可视化

## 安装依赖

```bash
pip install -r requirements.txt
```

确保安装了以下关键依赖：
- `wandb` - 实时性能监控
- `torch` - 深度学习框架
- `transformers` - Transformer模型
- `pandas` - 数据处理
- `numpy` - 数值计算
- `scikit-learn` - 机器学习工具
- `matplotlib` - 数据可视化
- `psutil` - 系统资源监控

## WandB配置

1. 创建WandB账户（如果尚未创建）：
   - 访问 [wandb.ai](https://wandb.ai) 注册账户
   - 获取API密钥

2. 登录WandB：
   ```bash
   wandb login
   ```
   输入您的API密钥

3. 配置WandB参数（在配置文件中）：
   ```yaml
   wandb:
     project_name: "stellar-mass-prediction"  # 项目名称
     entity: "your-username"  # 您的用户名或团队名
     tags: ["transformer", "stellar-physics"]  # 实验标签
     mode: "online"  # 运行模式
   ```

## 使用方法

### 1. 准备数据

确保您的数据文件位于正确的目录中：
- 训练数据：`data/train.csv`
- 验证数据：`data/val.csv`
- 测试数据：`data/test.csv`

数据应包含以下列：
- `logL`：对数光度
- `logTeff`：对数有效温度
- `FeH`：金属丰度
- `age`：年龄
- `mass`：质量（目标变量）

### 2. 配置参数

使用提供的配置文件 `config_wandb.yaml`，或根据需要修改参数：

```bash
cp config_wandb.yaml config.yaml
# 编辑config.yaml文件，修改必要的参数
```

### 3. 运行训练

使用以下命令启动训练：

```bash
python train.py --config config.yaml
```

训练过程中，WandB将自动记录以下信息：
- 训练和验证损失
- R²分数
- 学习率变化
- 系统资源使用情况
- 模型性能指标

### 4. 查看实时监控

训练开始后，您可以通过以下方式查看实时监控：

1. **终端链接**：训练启动后，终端会显示WandB仪表板的链接
2. **WandB网站**：访问 [wandb.ai](https://wandb.ai) 查看您的项目
3. **实时图表**：查看训练损失、验证损失、R²分数等实时变化
4. **系统监控**：查看CPU、内存和GPU使用情况
5. **超参数比较**：比较不同实验的超参数和结果

## 监控功能详解

### 实时性能指标

- **训练损失**：每个epoch的平均训练损失
- **验证损失**：每个epoch的验证损失
- **R²分数**：模型在训练和验证集上的R²分数
- **学习率**：当前学习率（如果使用学习率调度）
- **梯度范数**：梯度裁剪前的梯度范数

### 系统资源监控

- **CPU使用率**：CPU使用百分比
- **内存使用率**：内存使用百分比
- **GPU使用率**：GPU使用百分比（如果使用GPU）
- **GPU内存使用率**：GPU内存使用百分比（如果使用GPU）

### 模型检查点

- 每个epoch的模型性能
- 最佳模型保存
- 模型架构可视化
- 模型参数分布

### 自定义图表

- 预测结果散点图
- 残差分布图
- 特征重要性图
- 学习率调度图

## 高级功能

### 1. 自定义指标

您可以在训练代码中添加自定义指标：

```python
# 在train.py中
wandb.log({
    "custom_metric_1": value1,
    "custom_metric_2": value2
})
```

### 2. 模型版本控制

WandB自动保存模型检查点，您可以轻松比较不同版本的模型：

```python
# 保存模型
wandb.save("model_checkpoint.pth")

# 恢复模型
run = wandb.init(project="stellar-mass-prediction")
wandb.restore("model_checkpoint.pth")
```

### 3. 超参数搜索

使用WandB的 sweeps 功能进行超参数搜索：

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.01},
        "batch_size": {"values": [16, 32, 64]},
        "hidden_dim": {"values": [64, 128, 256]}
    }
}
```

### 4. 数据版本控制

使用WandB Artifacts 进行数据版本控制：

```python
# 上传数据集
artifact = wandb.Artifact('stellar-dataset', type='dataset')
artifact.add_file('data/train.csv')
wandb.log_artifact(artifact)

# 使用数据集
artifact = run.use_artifact('stellar-dataset:latest')
dataset_dir = artifact.download()
```

## 故障排除

### 常见问题

1. **WandB连接问题**
   - 检查网络连接
   - 确认API密钥正确
   - 尝试离线模式：`mode: "offline"`

2. **系统监控不工作**
   - 确认安装了`psutil`库
   - 检查权限设置
   - 尝试降低更新频率

3. **GPU监控不显示**
   - 确认安装了`pynvml`库
   - 检查NVIDIA驱动程序
   - 确认GPU可用

### 调试模式

启用详细日志以进行调试：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 示例仪表板

训练过程中，您将看到类似以下的仪表板：

1. **概览页面**：显示主要指标和图表
2. **图表页面**：详细的训练和验证指标图表
3. **系统页面**：CPU、内存和GPU使用情况
4. **超参数页面**：当前实验的超参数配置
5. **历史页面**：所有实验的历史记录和比较

## 贡献

欢迎提交问题报告和功能请求！如果您想贡献代码，请：

1. Fork 本仓库
2. 创建功能分支
3. 提交您的更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。