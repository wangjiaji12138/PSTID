#!/bin/bash

# 模型列表
MODELS=(
    # "GRU"
    # "HA"
    "DCRNN"        # 2018, 图卷积开山作
    # 图学习（3个）
    "MTGNN"        # 2020, 自适应图
    "AGCRN"        # 2020, 节点自适应
    "GWNET"        # 2020, 自适应邻接矩阵
    # 动态图/解耦（1个）
    "STDN"         # 2025, 季节分解
    # 注意力/Transformer（1个）
    "STAEformer"   # 2023, 时空注意力
    # "PDFormer"
    # 元学习/强基线（1个）
    "MegaCRN"      # 2023, 元学习
    # 简单高效（2个）
    "STID"         # 2022, 纯 MLP
    "STNorm"       # 2021, 归一化
    # 对比学习（1个）
    "STSSL"
    "STSSDL"       # 2025, 自监督
)

# 数据集列表
DATASETS=(
    "cq"
    "hz"
    "jl"
    "sh"
    "yt"
    "nyc_taxi"
    "chi_taxi"
)

# 记录开始时间
START_TIME=$(date +%s)

for data in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "========================================"
        echo "Running: python train.py --data $data --model $model"
        echo "========================================"
        
        python train.py --data "$data" --model "$model" --epoch 100 --gpu 2
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[SUCCESS] $model on $data completed"
        else
            echo "[ERROR] $model on $data failed with exit code $EXIT_CODE"
        fi
        echo ""
    done
done

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================"
echo "All experiments completed!"
echo "Total time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
echo "========================================"
