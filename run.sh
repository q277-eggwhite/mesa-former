#!/bin/bash

# 恒星Transformer模型训练与评估脚本
# 作者: 戚泽清

echo "========================================"
echo "恒星Transformer模型训练与评估脚本"
echo "========================================"

# 检查Python环境
echo "[STEP] 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 未安装，请先安装Python3"
    exit 1
fi

# 创建必要目录（如果不存在）
echo "[STEP] 确保必要目录存在..."
mkdir -p models evaluation_results

# 训练模型
echo "[STEP] 开始训练模型..."
# 从配置文件读取world_size参数（忽略注释）
WORLD_SIZE=$(grep -E '^[[:space:]]*world_size:' config.yaml | awk -F: '{print $2}' | sed 's/#.*//' | tr -d '[:space:]' | tr -d '"')
echo "从配置文件读取world_size: $WORLD_SIZE"
# 使用torchrun启动分布式训练，使用配置文件中定义的卡数
torchrun --nproc_per_node=$WORLD_SIZE --master_port=12355 main.py

# 测试模型并生成可视化报告 - 只有主进程执行
if [ "$RANK" = "0" ] || [ -z "$RANK" ]; then
    echo "[STEP] 测试模型并生成可视化报告..."
    python3 test.py 
fi

# 显示结果摘要
echo "[STEP] 生成结果摘要..."
echo "- 模型文件: models/best_model.pth"
echo "- 评估结果: evaluation_results/"

echo ""
echo "========================================"
echo "脚本执行完成!"
echo "========================================"
