## 五个数据集的预测性能比较

对比 CQ、HZ、NYC-Taxi、CHI-Taxi、NYC-Bike，展示 MAE、RMSE 和 MAPE（%），仅报告均值。

统计口径：

- 数据仅取自 `results_all/<数据集>/<模型>_<seed>_<时间戳及后缀>/test_results.json`，不纳入 `results` 下正在补跑的其他实验。
- 五个数据集、14 个模型均覆盖 seed=13、42、88，共使用 210 个完整模型实验；每个数据集、模型、seed 仅有一份结果。
- HimNet 的 15 个实验统一使用 `num_layers=1`、`input_embedding_dim=16`、`use_teacher_forcing=0`、`tf_decay_steps=2`；数据集对应的时间槽允许不同。
- 展示三个种子的等权算术均值，不展示标准差，保留三位小数；先用原始精度计算和排名，再四舍五入。
- 模型名忽略大小写；PROPHET 的旧配置模型名 `PSTID` 统一归入 PROPHET。PROPHET 仅纳入完整模型，排除关闭 Proto、Spatio、Temporal、Adj、His 的消融及旧版 `wo_P_LinearSoftmaxLinear`。`proto_ablation=direct_embeddings_mlp` 是当前配置的默认字段，`use_proto=true` 时不据此判定为消融；旧配置未保存 `use_his` 时按默认开启处理。
- 按 `config.json` 中的实际 seed 核对，不依据历史所属目录推断。
- MAPE 直接使用测试 JSON 中的百分数值，不再乘以 100。
- 各指标越低越好；按未舍入的均值标出最优（粗体）和次优（下划线）。
- 提升百分比以每个数据集、每项指标的最优基线（排除 PROPHET）为参照，按 `(最优基线均值 − PROPHET 均值) / 最优基线均值 × 100%` 计算；使用未舍入的三种子均值，最终保留两位小数。正值表示改善，负值表示退步。
- LaTeX 导言区需加载 `booktabs`、`multirow` 和 `graphicx`；使用 `table*` 跨双栏排版，并按正文宽度缩放。

```tex
\subsection{Comparisons with Baselines}
We compare PROPHET with 13 representative baselines spanning statistical forecasting, graph-based modeling, normalization, embedding-based methods, and pattern learning:
\textbf{HA}~\cite{ha}, \textbf{DCRNN}~\cite{dcrnn}, \textbf{STNorm}~\cite{stnorm}, \textbf{MTGNN}~\cite{mtgnn}, \textbf{AGCRN}~\cite{agcrn}, \textbf{GWNet}~\cite{gwnet}, \textbf{STDN}~\cite{stdn}, \textbf{STID}~\cite{stid}, \textbf{STAEformer}~\cite{staeformer}, \textbf{MegaCRN}~\cite{megacrn}, \textbf{ST-SSL}~\cite{stssl}, \textbf{HimNet}~\cite{himnet}, and \textbf{ST-SSDL}~\cite{stssdl}.
We report mean absolute error (MAE) and root mean squared error (RMSE) in demand units, and mean absolute percentage error (MAPE). MAE and RMSE include zero-demand targets, while MAPE is computed over targets with demand at least 0.5.
Table~\ref{comparisons} reports the arithmetic mean over three random seeds (13, 42, and 88) for every model on all five datasets.
PROPHET achieves the lowest MAE and RMSE on all five datasets and the lowest MAPE on CQ, HZ, CHI-Taxi, and NYC-Bike, ranking first in 14 of the 15 dataset--metric comparisons.
The largest relative improvements are observed on NYC-Bike, where PROPHET reduces MAE, RMSE, and MAPE by 18.83\%, 19.28\%, and 20.05\%, respectively, relative to the best baseline for each metric.
On NYC-Taxi, PROPHET reduces MAE and RMSE by 3.69\% and 5.43\%, respectively, relative to MTGNN, but ranks third in MAPE behind MegaCRN and MTGNN; its MAPE is 1.31\% higher than that of MegaCRN.

\begin{table*}[t]
\centering
\caption{Forecasting performance on CQ, HZ, NYC-Taxi, CHI-Taxi, and NYC-Bike. Results are averaged over three different seeds. Lower values are better. The best and second-best means are shown in bold and underlined, respectively. Improvement is the relative error reduction of PROPHET over the best baseline for each metric; negative values indicate worse performance.}
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

PROPHET 完整模型三种子结果来源：

- cq seed=13: `results_all/cq/PROPHET_13_20261016_052959/test_results.json`
- cq seed=42: `results_all/cq/PROPHET_42_20261010_045203/test_results.json`
- cq seed=88: `results_all/cq/PROPHET_88_20260920_215035__Full/test_results.json`
- hz seed=13: `results_all/hz/PROPHET_13_20261016_092059/test_results.json`
- hz seed=42: `results_all/hz/PROPHET_42_20261010_083451/test_results.json`
- hz seed=88: `results_all/hz/PROPHET_88_20260920_215923__Full/test_results.json`
- nyc_taxi seed=13: `results_all/nyc_taxi/PROPHET_13_20261017_215925/test_results.json`
- nyc_taxi seed=42: `results_all/nyc_taxi/PROPHET_42_20261011_200251/test_results.json`
- nyc_taxi seed=88: `results_all/nyc_taxi/PROPHET_88_20260920_225812__Full/test_results.json`
- chi_taxi seed=13: `results_all/chi_taxi/PROPHET_13_20261018_082812/test_results.json`
- chi_taxi seed=42: `results_all/chi_taxi/PROPHET_42_20261012_063407/test_results.json`
- chi_taxi seed=88: `results_all/chi_taxi/PROPHET_88_20260920_223757__Full/test_results.json`
- nyc_bike seed=13: `results_all/nyc_bike/PROPHET_13_20261020_230441/test_results.json`
- nyc_bike seed=42: `results_all/nyc_bike/PROPHET_42_20261021_132813/test_results.json`
- nyc_bike seed=88: `results_all/nyc_bike/PROPHET_88_20261002_022630__Full/test_results.json`

HimNet 三种子结果来源（均为统一配置的新实验）：

- cq seed=13: `results_all/cq/HIMNET_13_20261020_210555/test_results.json`
- cq seed=42: `results_all/cq/HIMNET_42_20261021_041630/test_results.json`
- cq seed=88: `results_all/cq/HIMNET_88_20261021_111550/test_results.json`
- hz seed=13: `results_all/hz/HIMNET_13_20261020_213746/test_results.json`
- hz seed=42: `results_all/hz/HIMNET_42_20261021_043903/test_results.json`
- hz seed=88: `results_all/hz/HIMNET_88_20261021_114212/test_results.json`
- nyc_taxi seed=13: `results_all/nyc_taxi/HIMNET_13_20261020_220341/test_results.json`
- nyc_taxi seed=42: `results_all/nyc_taxi/HIMNET_42_20261021_050134/test_results.json`
- nyc_taxi seed=88: `results_all/nyc_taxi/HIMNET_88_20261021_120725/test_results.json`
- chi_taxi seed=13: `results_all/chi_taxi/HIMNET_13_20261021_013400/test_results.json`
- chi_taxi seed=42: `results_all/chi_taxi/HIMNET_42_20261021_082317/test_results.json`
- chi_taxi seed=88: `results_all/chi_taxi/HIMNET_88_20261021_152941/test_results.json`
- nyc_bike seed=13: `results_all/nyc_bike/HIMNET_13_20261021_025257/test_results.json`
- nyc_bike seed=42: `results_all/nyc_bike/HIMNET_42_20261021_095051/test_results.json`
- nyc_bike seed=88: `results_all/nyc_bike/HIMNET_88_20261021_164944/test_results.json`

提升百分比对应的最优基线（依次为 MAE / RMSE / MAPE）：

| 数据集 | 最优基线（MAE / RMSE / MAPE） | MAE 提升 | RMSE 提升 | MAPE 提升 |
|---|---|---:|---:|---:|
| cq | STID / STID / STID | 2.94% | 4.18% | 2.00% |
| hz | STID / STID / STID | 3.30% | 5.01% | 2.26% |
| nyc_taxi | MTGNN / MTGNN / MegaCRN | 3.69% | 5.43% | -1.31% |
| chi_taxi | MTGNN / MTGNN / ST-SSDL | 6.56% | 10.91% | 0.53% |
| nyc_bike | MTGNN / MTGNN / ST-SSDL | 18.83% | 19.28% | 20.05% |
