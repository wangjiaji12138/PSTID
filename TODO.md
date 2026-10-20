## 五个数据集的预测性能比较

对比 CQ、HZ、NYC-Taxi、CHI-Taxi、NYC-Bike，展示 MAE、RMSE 和 MAPE（%），仅报告均值。

统计口径：

- 数据取自 `results_88`（seed=88，原 `results` 目录）、`results_13`（seed=13）、`results_42`（seed=42）的 `test_results.json`；原要求中的 `result_13` 更正为实际目录名 `results_13`。
- 前四个数据集的每个模型均使用三个种子的结果，展示三个种子的等权算术均值，不展示标准差，保留三位小数；先用原始精度计算和排名，再四舍五入。
- NYC-Bike 在 `results_13` 和 `results_42` 中没有结果。该列暂列 seed=88 的单次结果并加 `†`，不作为三种子均值；补齐两个种子后才能完成三种子统计。
- 模型名忽略大小写，旧名 PSTID 的完整模型归入 PROPHET；排除关闭 Proto、Spatio、Temporal、Adj、His 或指定 `proto_ablation` 的消融实验。旧配置未保存 `use_his` 时按默认开启处理。
- 按 `config.json` 中的实际种子筛选，排除 `results_88` 中混入的 seed=13 实验。
- MAPE 直接使用测试 JSON 中的百分数值，不再乘以 100。
- 各指标越低越好；按未舍入的均值标出最优（粗体）和次优（下划线），NYC-Bike 按单次结果排名。
- LaTeX 导言区需加载 `booktabs`、`multirow` 和 `graphicx`；使用 `table*` 跨双栏排版，并按正文宽度缩放。

```tex
\begin{table*}[t]
\centering
\caption{Forecasting performance on CQ, HZ, NYC-Taxi, CHI-Taxi, and NYC-Bike. Results are averaged over seeds 88, 13, and 42, except for NYC-Bike ($\dagger$), where only seed 88 is available. Lower values are better. The best and second-best means (single-run values for NYC-Bike) are shown in bold and underlined, respectively.}
\label{comparisons}
\begingroup
\fontsize{8.5}{10.5}\selectfont
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | ccc | ccc | ccc | ccc | ccc}
\toprule
\multirow{2}{*}{Models} & \multicolumn{3}{c}{CQ} & \multicolumn{3}{c}{HZ} & \multicolumn{3}{c}{NYC-Taxi} & \multicolumn{3}{c}{CHI-Taxi} & \multicolumn{3}{c}{NYC-Bike$^{\dagger}$} \\
& MAE & RMSE & MAPE (\%) & MAE & RMSE & MAPE (\%) & MAE & RMSE & MAPE (\%) & MAE & RMSE & MAPE (\%) & MAE & RMSE & MAPE (\%) \\
\midrule
HA         & $1.325$ & $3.577$ & $78.834$ & $2.749$ & $6.273$ & $89.984$ & $17.146$ & $45.911$ & $147.039$ & $3.252$ & $10.886$ & $115.351$ & $9.060$ & $18.122$ & $251.038$ \\
DCRNN      & $0.660$ & $1.618$ & $53.884$ & $1.283$ & $2.801$ & $48.777$ & $5.113$ & $15.115$ & $47.595$ & $1.289$ & $3.589$ & $61.085$ & $3.684$ & $8.443$ & $81.030$ \\
STNorm     & $0.797$ & $2.001$ & $57.230$ & $1.741$ & $4.432$ & $55.240$ & $5.763$ & $16.902$ & $52.208$ & $1.548$ & $5.223$ & $64.827$ & $5.723$ & $12.045$ & $154.087$ \\
MTGNN      & $0.671$ & $1.634$ & $51.708$ & $1.216$ & $2.660$ & $46.906$ & $4.242$ & $12.030$ & $44.473$ & $\underline{1.247}$ & $\underline{3.521}$ & $60.330$ & $3.457$ & $7.718$ & $74.269$ \\
AGCRN      & $1.068$ & $3.531$ & $66.055$ & $1.873$ & $5.711$ & $56.629$ & $5.237$ & $16.005$ & $49.302$ & $1.832$ & $7.857$ & $63.983$ & $4.231$ & $10.302$ & $88.503$ \\
GWNet      & $0.656$ & $1.639$ & $52.942$ & $1.285$ & $3.061$ & $47.664$ & $4.885$ & $14.141$ & $48.839$ & $1.386$ & $4.372$ & $61.261$ & $5.144$ & $11.078$ & $136.256$ \\
STDN       & $0.653$ & $1.661$ & $54.320$ & $1.242$ & $3.057$ & $46.099$ & $6.740$ & $18.746$ & $70.921$ & $1.987$ & $7.687$ & $67.491$ & $5.107$ & $11.451$ & $118.189$ \\
STID       & $\underline{0.649}$ & $\underline{1.580}$ & $\underline{51.474}$ & $\underline{1.154}$ & $\underline{2.419}$ & $\underline{44.181}$ & $5.249$ & $15.379$ & $49.638$ & $1.310$ & $3.858$ & $60.626$ & $3.722$ & $8.606$ & $78.243$ \\
STAEformer & $0.656$ & $1.607$ & $52.765$ & $1.219$ & $2.752$ & $46.368$ & $4.805$ & $13.879$ & $50.114$ & $1.288$ & $3.745$ & $61.166$ & $3.466$ & $7.635$ & $75.825$ \\
ST-SSL     & $0.748$ & $1.897$ & $58.427$ & $1.417$ & $3.353$ & $49.992$ & $5.530$ & $16.110$ & $50.822$ & $1.480$ & $4.764$ & $67.971$ & $4.427$ & $10.048$ & $99.780$ \\
MegaCRN    & $0.694$ & $1.734$ & $54.016$ & $1.316$ & $3.164$ & $48.152$ & $4.248$ & $12.254$ & $\underline{44.010}$ & $1.358$ & $4.127$ & $61.177$ & $3.604$ & $8.389$ & $75.577$ \\
HimNet     & $0.690$ & $1.808$ & $53.688$ & $1.299$ & $3.267$ & $49.020$ & $\mathbf{3.917}$ & $\mathbf{11.114}$ & $\mathbf{42.741}$ & $1.318$ & $4.204$ & $59.268$ & $\underline{3.356}$ & $\underline{7.488}$ & $\underline{71.989}$ \\
ST-SSDL    & $0.760$ & $2.179$ & $54.028$ & $1.434$ & $4.234$ & $50.042$ & $9.834$ & $33.939$ & $58.006$ & $1.495$ & $5.465$ & $\underline{59.121}$ & $3.593$ & $8.371$ & $75.156$ \\
PROPHET    & $\mathbf{0.630}$ & $\mathbf{1.514}$ & $\mathbf{50.444}$ & $\mathbf{1.116}$ & $\mathbf{2.297}$ & $\mathbf{43.184}$ & $\underline{4.085}$ & $\underline{11.376}$ & $44.588$ & $\mathbf{1.166}$ & $\mathbf{3.137}$ & $\mathbf{58.808}$ & $\mathbf{2.841}$ & $\mathbf{6.284}$ & $\mathbf{60.470}$ \\
\bottomrule
\end{tabular}%
}
\endgroup
\par\smallskip
\begin{minipage}{\textwidth}
\footnotesize $\dagger$ NYC-Bike reports single-run results for seed 88; results for seeds 13 and 42 are unavailable.
\end{minipage}
\end{table*}
```

PROPHET 完整模型的 seed=88 来源（其余两个种子使用各自目录下的 `PROPHET_*/test_results.json`）：

- cq: `results_88/cq/PSTID_20260920_215035/test_results.json`
- hz: `results_88/hz/PSTID_20260920_215923/test_results.json`
- nyc_taxi: `results_88/nyc_taxi/PSTID_20260920_225812/test_results.json`
- chi_taxi: `results_88/chi_taxi/PSTID_20260920_223757/test_results.json`
- nyc_bike: `results_88/nyc_bike/PSTID_20261002_022630/test_results.json`
