#!/bin/bash

# 模型列表
MODELS=(
    "HA"
    "DCRNN"
    "AGCRN"
    "MTGNN"
    "GWNET"
    "STDN"
    "STAEformer" 
    "MegaCRN"
    "STID"
    "STNorm"
    "STSSL"
    "STSSDL"
    # "HimNet"
    "PROPHET"
)


# 数据集列表
DATASETS=(
    # "cq"
    # "hz"
    "nyc_taxi"
    # "chi_taxi"
    # "nyc_bike"
)


# 记录开始时间
START_TIME=$(date +%s)

SEED=(
    "13"
    "42"
    "88"
)

for seed in "${SEED[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "========================================"
        echo "Running: python train.py --seed $seed --model $model --data nyc_taxi"
        echo "========================================"
        
        python train.py --seed $seed --model "$model" --epoch 100 --gpu 4 --data nyc_taxi --input_embedding_dim 16 --num_layers 1
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[SUCCESS] $model on $seed completed"
        else
            echo "[ERROR] $model on $seed failed with exit code $EXIT_CODE"
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
