#!/bin/bash

# ============================================================
# 消融实验脚本
# ============================================================  
# 使用方法:
#   bash run_abl.sh     
# ============================================================

MODEL=${1:-prophet}
MODEL=${MODEL,,}

# 消融实验开关列表
ABL=(
    "use_proto"
    "use_spatio"
    "use_temporal"
    "use_adj"
    "use_his"
)

# 数据集列表
DATASETS=(
    "cq"
    "hz"
    "chi_taxi"
    "nyc_taxi"
    "nyc_bike"
)

GPU=2
EPOCHS=100

# 验证模型名称
if [[ "$MODEL" != "prophet" ]]; then
    echo "[ERROR] 未知模型: $MODEL"
    echo "使用方法: bash run_abl.sh [prophet]"
    exit 1
fi

MODEL_UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')

# 记录开始时间
START_TIME=$(date +%s)

for data in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Running: python train.py --data $data --model $MODEL_UPPER (Full)"
    echo "========================================"

    python train.py --data $data --model $MODEL_UPPER --epochs $EPOCHS --gpu $GPU
    FULL_EXIT=$?

    if [ $FULL_EXIT -ne 0 ]; then
        echo "[ERROR] Full model on $data failed with exit code $FULL_EXIT, skipping ablations"
        continue
    fi
    echo "[SUCCESS] Full model on $data completed"

    for abl in "${ABL[@]}"; do
        echo "----------------------------------------"
        echo "Ablation: $abl = 0"
        echo "----------------------------------------"

        python train.py --data $data --model $MODEL_UPPER --$abl 0 --epochs $EPOCHS --gpu $GPU --results_dir results/

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[SUCCESS] $data w/o $abl completed"
        else
            echo "[ERROR] $data w/o $abl failed with exit code $EXIT_CODE"
        fi
        echo ""
    done
done

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================"
echo "All experiments completed!"
echo "Model: $MODEL_UPPER"
echo "Total time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
echo "========================================"
echo ""
echo "运行以下命令分析结果:"
echo "  python -m experiments.ablation"
