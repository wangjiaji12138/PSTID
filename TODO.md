## 五个数据集的预测性能比较

对比 CQ、HZ、NYC-Taxi、CHI-Taxi、NYC-Bike，展示 MAE、RMSE 和 MAPE（%），仅报告均值。

统计口径：

- 数据取自 `results_88`（seed=88，原 `results` 目录）、`results_13`（seed=13）、`results_42`（seed=42）的 `test_results.json`；原要求中的 `result_13` 更正为实际目录名 `results_13`。
- HimNet 在五个数据集上的三个种子（13、42、88）均取自 `results/<数据集>/HIMNET_*/test_results.json`；统一使用 `num_layers=1`、`input_embedding_dim=16`、`use_teacher_forcing=0`、`tf_decay_steps=2` 的配置。
- 除 HimNet 外，NYC-Bike 的 seed=13、42 结果取自 `results/nyc_bike/*/test_results.json`，seed=88 取自 `results_88/nyc_bike`。
- 五个数据集的每个模型均使用三个种子的结果，展示三个种子的等权算术均值，不展示标准差，保留三位小数；先用原始精度计算和排名，再四舍五入。
- 模型名忽略大小写，旧名 PSTID 的完整模型归入 PROPHET；排除关闭 Proto、Spatio、Temporal、Adj、His 的消融实验；`proto_ablation=direct_embeddings_mlp` 是当前配置的默认字段，`use_proto=true` 时不据此判定为消融。旧配置未保存 `use_his` 时按默认开启处理。
- 按 `config.json` 中的实际种子筛选，排除 `results_88` 中混入的 seed=13 实验。
- MAPE 直接使用测试 JSON 中的百分数值，不再乘以 100。
- 各指标越低越好；按未舍入的均值标出最优（粗体）和次优（下划线）。
- 提升百分比以每个数据集、每项指标的最优基线（排除 PROPHET）为参照，按 `(最优基线均值 − PROPHET 均值) / 最优基线均值 × 100%` 计算；使用未舍入的三种子均值，最终保留两位小数。正值表示改善，负值表示退步。
- LaTeX 导言区需加载 `booktabs`、`multirow` 和 `graphicx`；使用 `table*` 跨双栏排版，并按正文宽度缩放。

```tex
\begin{table*}[t]
\centering
\caption{Forecasting performance on CQ, HZ, NYC-Taxi, CHI-Taxi, and NYC-Bike. Results are averaged over seeds 88, 13, and 42. Lower values are better. The best and second-best means are shown in bold and underlined, respectively. Improvement is the relative error reduction of PROPHET over the best baseline for each metric; negative values indicate worse performance.}
\label{comparisons}
\begingroup
\fontsize{8.5}{10.5}\selectfont
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | ccc | ccc | ccc | ccc | ccc}
\toprule
\multirow{2}{*}{Models} & \multicolumn{3}{c}{CQ} & \multicolumn{3}{c}{HZ} & \multicolumn{3}{c}{NYC-Taxi} & \multicolumn{3}{c}{CHI-Taxi} & \multicolumn{3}{c}{NYC-Bike} \\
& MAE & RMSE & MAPE (\%) & MAE & RMSE & MAPE (\%) & MAE & RMSE & MAPE (\%) & MAE & RMSE & MAPE (\%) & MAE & RMSE & MAPE (\%) \\
\midrule
HA         & $1.325$ & $3.577$ & $78.834$ & $2.749$ & $6.273$ & $89.984$ & $17.146$ & $45.911$ & $147.039$ & $3.252$ & $10.886$ & $115.351$ & $9.060$ & $18.122$ & $251.038$ \\
DCRNN      & $0.660$ & $1.618$ & $53.884$ & $1.283$ & $2.801$ & $48.777$ & $5.113$ & $15.115$ & $47.595$ & $1.289$ & $3.589$ & $61.085$ & $4.194$ & $9.080$ & $100.660$ \\
STNorm     & $0.797$ & $2.001$ & $57.230$ & $1.741$ & $4.432$ & $55.240$ & $5.763$ & $16.902$ & $52.208$ & $1.548$ & $5.223$ & $64.827$ & $5.870$ & $12.512$ & $156.443$ \\
MTGNN      & $0.671$ & $1.634$ & $51.708$ & $1.216$ & $2.660$ & $46.906$ & $\underline{4.242}$ & $\underline{12.030}$ & $\underline{44.473}$ & $\underline{1.247}$ & $\underline{3.521}$ & $60.330$ & $\underline{3.500}$ & $\underline{7.785}$ & $75.576$ \\
AGCRN      & $1.068$ & $3.531$ & $66.055$ & $1.873$ & $5.711$ & $56.629$ & $5.237$ & $16.005$ & $49.302$ & $1.832$ & $7.857$ & $63.983$ & $4.256$ & $10.370$ & $88.830$ \\
GWNet      & $0.656$ & $1.639$ & $52.942$ & $1.285$ & $3.061$ & $47.664$ & $4.885$ & $14.141$ & $48.839$ & $1.386$ & $4.372$ & $61.261$ & $5.349$ & $11.318$ & $142.796$ \\
STDN       & $0.653$ & $1.661$ & $54.320$ & $1.242$ & $3.057$ & $46.099$ & $6.740$ & $18.746$ & $70.921$ & $1.987$ & $7.687$ & $67.491$ & $5.138$ & $11.379$ & $119.726$ \\
STID       & $\underline{0.649}$ & $\underline{1.580}$ & $\underline{51.474}$ & $\underline{1.154}$ & $\underline{2.419}$ & $\underline{44.181}$ & $5.249$ & $15.379$ & $49.638$ & $1.310$ & $3.858$ & $60.626$ & $3.723$ & $8.612$ & $78.708$ \\
STAEformer & $0.656$ & $1.607$ & $52.765$ & $1.219$ & $2.752$ & $46.368$ & $4.805$ & $13.879$ & $50.114$ & $1.288$ & $3.745$ & $61.166$ & $3.546$ & $7.796$ & $77.394$ \\
ST-SSL     & $0.748$ & $1.897$ & $58.427$ & $1.417$ & $3.353$ & $49.992$ & $5.530$ & $16.110$ & $50.822$ & $1.480$ & $4.764$ & $67.971$ & $4.348$ & $9.911$ & $96.971$ \\
MegaCRN    & $0.694$ & $1.734$ & $54.016$ & $1.316$ & $3.164$ & $48.152$ & $4.248$ & $12.254$ & $\mathbf{44.010}$ & $1.358$ & $4.127$ & $61.177$ & $3.635$ & $8.423$ & $77.240$ \\
HimNet     & $0.818$ & $2.583$ & $56.013$ & $1.528$ & $4.529$ & $50.145$ & $4.727$ & $13.933$ & $46.462$ & $1.488$ & $5.532$ & $61.187$ & $3.911$ & $8.918$ & $84.780$ \\
ST-SSDL    & $0.760$ & $2.179$ & $54.028$ & $1.434$ & $4.234$ & $50.042$ & $9.834$ & $33.939$ & $58.006$ & $1.495$ & $5.465$ & $\underline{59.121}$ & $3.565$ & $8.290$ & $\underline{75.389}$ \\
\midrule
PROPHET    & $\mathbf{0.630}$ & $\mathbf{1.514}$ & $\mathbf{50.444}$ & $\mathbf{1.116}$ & $\mathbf{2.297}$ & $\mathbf{43.184}$ & $\mathbf{4.085}$ & $\mathbf{11.376}$ & $44.588$ & $\mathbf{1.166}$ & $\mathbf{3.137}$ & $\mathbf{58.808}$ & $\mathbf{2.841}$ & $\mathbf{6.285}$ & $\mathbf{60.275}$ \\
Improvement (\%) & $2.94\%$ & $4.18\%$ & $2.00\%$ & $3.30\%$ & $5.01\%$ & $2.26\%$ & $3.69\%$ & $5.43\%$ & $-1.31\%$ & $6.56\%$ & $10.91\%$ & $0.53\%$ & $18.83\%$ & $19.28\%$ & $20.05\%$ \\
\bottomrule
\end{tabular}%
}
\endgroup
\end{table*}
```

PROPHET 完整模型的 seed=88 来源（其余两个种子中，前四个数据集使用各自种子目录、NYC-Bike 使用 `results/nyc_bike` 下的 `PROPHET_*/test_results.json`）：

- cq: `results_88/cq/PSTID_20260920_215035/test_results.json`
- hz: `results_88/hz/PSTID_20260920_215923/test_results.json`
- nyc_taxi: `results_88/nyc_taxi/PSTID_20260920_225812/test_results.json`
- chi_taxi: `results_88/chi_taxi/PSTID_20260920_223757/test_results.json`
- nyc_bike: `results_88/nyc_bike/PROPHET_20261002_022630__Full/test_results.json`

HimNet 三种子结果来源（旧配置和空目录已清理）：

- cq seed=13: `results/cq/HIMNET_20261020_210555/test_results.json`
- cq seed=42: `results/cq/HIMNET_20261021_041630/test_results.json`
- cq seed=88: `results/cq/HIMNET_20261021_111550/test_results.json`
- hz seed=13: `results/hz/HIMNET_20261020_213746/test_results.json`
- hz seed=42: `results/hz/HIMNET_20261021_043903/test_results.json`
- hz seed=88: `results/hz/HIMNET_20261021_114212/test_results.json`
- nyc_taxi seed=13: `results/nyc_taxi/HIMNET_20261020_220341/test_results.json`
- nyc_taxi seed=42: `results/nyc_taxi/HIMNET_20261021_050134/test_results.json`
- nyc_taxi seed=88: `results/nyc_taxi/HIMNET_20261021_120725/test_results.json`
- chi_taxi seed=13: `results/chi_taxi/HIMNET_20261021_013400/test_results.json`
- chi_taxi seed=42: `results/chi_taxi/HIMNET_20261021_082317/test_results.json`
- chi_taxi seed=88: `results/chi_taxi/HIMNET_20261021_152941/test_results.json`
- nyc_bike seed=13: `results/nyc_bike/HIMNET_20261021_025257/test_results.json`
- nyc_bike seed=42: `results/nyc_bike/HIMNET_20261021_095051/test_results.json`
- nyc_bike seed=88: `results/nyc_bike/HIMNET_20261021_164944/test_results.json`

NYC-Bike 新增 PROPHET 完整模型的来源：
- PROPHET seed=13: `results/nyc_bike/PROPHET_20261020_230441/test_results.json`
- PROPHET seed=42: `results/nyc_bike/PROPHET_20261021_132813/test_results.json`

提升百分比对应的最优基线（依次为 MAE / RMSE / MAPE）：

| 数据集 | 最优基线（MAE / RMSE / MAPE） | MAE 提升 | RMSE 提升 | MAPE 提升 |
|---|---|---:|---:|---:|
| cq | STID / STID / STID | 2.94% | 4.18% | 2.00% |
| hz | STID / STID / STID | 3.30% | 5.01% | 2.26% |
| nyc_taxi | MTGNN / MTGNN / MegaCRN | 3.69% | 5.43% | -1.31% |
| chi_taxi | MTGNN / MTGNN / ST-SSDL | 6.56% | 10.91% | 0.53% |
| nyc_bike | MTGNN / MTGNN / ST-SSDL | 18.83% | 19.28% | 20.05% |
