#!/bin/bash
# Grid search for PSTID hyperparameters on SH dataset
# M experiment: fix proto_emb_dim=64, vary num_spatial_prototypes (4,8,12,16,20)
# d experiment: fix num_spatial_prototypes=16, vary proto_emb_dim (16,32,48,64,80)

DATA="chi_taxi"
GPU=1

echo "========================================"
echo "M Experiment: proto_emb_dim=64, num_spatial_prototypes in [4,8,12,16,20]"
echo "========================================"
for M in 4 8 12 16 20; do
    echo "Running: proto_emb_dim=64, num_spatial_prototypes=$M"
    python train.py --model PSTID --data $DATA --proto_emb_dim 64 --num_spatial_prototypes $M --gpu $GPU
done

echo ""
echo "========================================"
echo "d Experiment: num_spatial_prototypes=16, proto_emb_dim in [16,32,48,64,80]"
echo "========================================"
for D in 16 32 48 64 80; do
    echo "Running: num_spatial_prototypes=16, proto_emb_dim=$D"
    python train.py --model PSTID --data $DATA --proto_emb_dim $D --num_spatial_prototypes 16 --gpu $GPU
done

echo ""
echo "All experiments completed!"
