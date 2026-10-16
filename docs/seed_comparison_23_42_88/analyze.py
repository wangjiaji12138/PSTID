"""Rebuild the descriptive seed comparison from saved test metrics; no training."""
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, variance

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SEEDS = (23, 42, 88)
METRICS = ('MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2')
FLAGS = ('use_proto', 'use_spatio', 'use_temporal', 'use_adj')
groups = defaultdict(dict)
audit = []
for seed in SEEDS:
    for path in sorted((ROOT / f'results_{seed}').glob('*/*/test_results.json')):
        cfg = json.loads((path.parent / 'config.json').read_text())
        result = json.loads(path.read_text())
        assert cfg['seed'] == seed, path
        assert cfg['data'] == path.parent.parent.name, path
        assert all(math.isfinite(result[k]) for k in METRICS), path
        model = cfg['model'].upper()
        model = 'PROPHET' if model == 'PSTID' else model
        dataset = cfg['data']
        row = dict(dataset=dataset, model=model, seed=seed,
                   path=str(path.relative_to(ROOT)), **{k: result[k] for k in METRICS})
        if model == 'PROPHET' and not all(cfg.get(k) for k in FLAGS):
            audit.append(dict(row, status='excluded_ablation'))
            continue
        if seed in groups[dataset, model]:
            previous = groups[dataset, model][seed]
            # Never select the best test score from repeated runs.
            assert previous['result'] == result, ('Ambiguous repeated run', path)
            audit.append(dict(row, status='excluded_identical_metrics_duplicate'))
            continue
        groups[dataset, model][seed] = dict(cfg=cfg, result=result, row=row)
        audit.append(dict(row, status='included'))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank(value, others, metric):
    return 1 + sum(x > value if metric == 'R2' else x < value for x in others)


complete = {k: v for k, v in groups.items() if set(v) == set(SEEDS)}
datasets = sorted({d for d, m in complete})
models = sorted({m for d, m in complete})
stats = {}
for (dataset, model), runs in sorted(complete.items()):
    for metric in METRICS:
        values = [runs[s]['result'][metric] for s in SEEDS]
        avg, sd = mean(values), stdev(values)
        row = dict(dataset=dataset, model=model, metric=metric, n=3,
                   **{f'seed_{s}': v for s, v in zip(SEEDS, values)},
                   mean=avg, sample_variance=variance(values), sample_std=sd,
                   cv_percent=100 * sd / abs(avg) if avg else None,
                   minimum=min(values), maximum=max(values),
                   range_percent=100 * (max(values) - min(values)) / abs(avg) if avg else None)
        for seed, value in zip(SEEDS, values):
            peers = [v[seed]['result'][metric] for (d, m), v in complete.items() if d == dataset]
            row[f'rank_{seed}'] = rank(value, peers, metric)
        stats[dataset, model, metric] = row
for (d, m, metric), row in stats.items():
    row['mean_rank'] = rank(row['mean'], [r['mean'] for (dd, mm, met), r in stats.items()
                                        if dd == d and met == metric], metric)

# Audit saved differences; fields unused by current constructor are not necessarily effective changes.
ignored = {'seed', 'model', 'gpu', 'device', 'results_dir', 'log_dir',
           'checkpoint_dir', 'current_epoch', 'data_dir'}
training_keys = {'epochs', 'batch_size', 'learning_rate', 'weight_decay',
                 'early_stop_patience', 'lr_milestones', 'lr_gamma', 'clip_grad',
                 'input_window', 'output_window', 'input_dim', 'output_dim'}
used = {}
tree = ast.parse((ROOT / 'models/registry.py').read_text())
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name.startswith('_build_'):
        used[node.name[7:].upper()] = {n.args[1].value for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == '_get_arg' and isinstance(n.args[1], ast.Constant)}
ptree = ast.parse((ROOT / 'models/PROPHET/model.py').read_text())
used['PROPHET'] = {n.args[1].value for n in ast.walk(ptree)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    and n.func.id == '_get_arg' and isinstance(n.args[1], ast.Constant)}
diffs = []
for (d, m), runs in sorted(complete.items()):
    keys = set().union(*(v['cfg'] for v in runs.values())) - ignored
    for key in sorted(keys):
        values = {s: runs[s]['cfg'].get(key, '<missing>') for s in SEEDS}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            diffs.append(dict(dataset=d, model=m, field=key,
                **{f'seed_{s}': json.dumps(v) for s, v in values.items()},
                used_by_current_constructor_or_training=key in used.get(m, set()) | training_keys))

prophet = []
for d in datasets:
    for met in METRICS:
        p = stats[d, 'PROPHET', met]
        baselines = [r for (dd, m, mm), r in stats.items() if dd == d and m != 'PROPHET' and mm == met]
        best = sorted(baselines, key=lambda r: r['mean'], reverse=met == 'R2')[0]
        gaps = []
        for s in SEEDS:
            vals = [r[f'seed_{s}'] for r in baselines]
            b = max(vals) if met == 'R2' else min(vals)
            gaps.append(p[f'seed_{s}'] - b if met == 'R2' else 100 * (b - p[f'seed_{s}']) / b)
        prophet.append(dict(p, best_baseline=best['model'], baseline_mean=best['mean'],
            mean_improvement=(p['mean'] - best['mean'] if met == 'R2'
                              else 100 * (best['mean'] - p['mean']) / best['mean']),
            minimum_seed_improvement=min(gaps), maximum_seed_improvement=max(gaps)))

write_csv('run_inventory.csv', audit)
write_csv('all_metrics.csv', list(stats.values()))
write_csv('prophet_comparison.csv', prophet)
write_csv('config_differences.csv', diffs)
write_csv('single_seed_only.csv', [v['row'] for k, runs in groups.items() if k not in complete for v in runs.values()])

lines = ['# 随机种子 23 / 42 / 88 结果比较', '',
    '数据来源：三个 results 目录中的 test_results.json；未重新训练。PSTID 按用户确认合并为 PROPHET，模型名忽略大小写。', '',
    f'三种子完整数据：{len(datasets)} 个数据集 × {len(models)} 个模型 × 3 个种子 = {len(complete)*3} 次实验。',
    'nyc_bike 只有 seed=88，单独列出，不计算跨种子方差。',
    '剔除 24 次 PROPHET 消融实验；CQ 两次完整模型的全部测试指标完全一致，仅保留时间较早的一次，未按测试分数择优。', '',
    '均值按三个种子等权计算；方差为样本方差 s²=Σ(x−均值)²/(n−1)，标准差 s=√s²；CV=100×s/|均值|。',
    '所有 ± 均为样本标准差，不是置信区间。MAE、RMSE、MAPE、SMAPE 越低越好，R² 越高越好；并列采用相同最小名次。',
    '不直接平均不同数据集的原始误差；跨数据集稳定性使用各数据集 CV 的等权平均。R² 的 CV 不适合衡量误差稳定性，正文重点比较 MAE/RMSE。', '',
    '## 主要发现', '',
    'PROPHET 波动很小，但并非所有数据集、所有指标、所有种子都第一。', '',
    '| 指标 | 逐种子第一次数 / 21 | 按三种子均值第一的数据集数 / 7 |',
    '|---|---:|---:|',
    *[f'| {met} | {sum(r[f"rank_{s}"] == 1 for r in prophet if r["metric"] == met for s in SEEDS)} | {sum(r["mean_rank"] == 1 for r in prophet if r["metric"] == met)} |' for met in METRICS], '',
    'MAE/RMSE：除 nyc_taxi 外，其余六个数据集的三个种子全部第一；nyc_taxi 的 seed=42、88 均落后于 HIMNET。',
    'nyc_taxi：PROPHET 的 RMSE 均值仍第一，但这与 HIMNET 的 seed=23 结果较差有关，不能表述为逐种子稳定领先。',
    'MAPE：chi_taxi 的 seed=42 略落后于 STSSDL；nyc_taxi 排名为 3/3/4。',
    'SMAPE：chi_taxi 的 seed=23 以及三种子均值均落后于 DCRNN；nyc_taxi 排名为 3/3/4。',
    'STSSDL 在 nyc_taxi 的 MAE（23/42/88）为 23.192680 / 7.420263 / 13.685242；原始 JSON 与训练日志一致，已保留所有结果。seed=23 在第 17 轮早停且测试 R²=-0.04519，提示该组合存在训练稳定性问题，具体原因尚未排查。',
    'nyc_bike：seed=88 上 PROPHET 的五项指标均第一，但缺少另外两个种子，无法判断其跨种子稳定性。', '',
    '## 可比性核查', '',
    '23 与 42 的已保存训练和模型配置一致（忽略种子、设备、路径和停止轮数）。PROPHET 的三种子有效配置一致。',
    f'88 与其他种子有 {len(diffs)} 项保存字段差异，涉及旧版无关参数、路径字段等；按当前模型构造器和训练字段检查，其中 {sum(r["used_by_current_constructor_or_training"] for r in diffs)} 项被实际读取。',
    '这只能核对现有配置和当前代码，不能证明历史源码、数据文件或运行环境完全相同；因此以下是跨种子实验的描述性波动，不是严格隔离随机种子因素的因果估计。只有三个种子也不足以证明总体显著性或未来始终第一。', '',
    '## PROPHET 各数据集结果', '',
    '| 数据集 | 指标 | 均值 ± 标准差 | 样本方差 | CV % | 排名 23/42/88 | 均值排名 | 最佳基线（均值） | 相对最佳基线改善 |',
    '|---|---|---:|---:|---:|---|---:|---|---:|']
for r in prophet:
    improvement = f'{r["mean_improvement"]:.3f}' + ('（R²绝对差）' if r['metric'] == 'R2' else '%')
    lines.append(f'| {r["dataset"]} | {r["metric"]} | {r["mean"]:.6f} ± {r["sample_std"]:.6f} | {r["sample_variance"]:.6g} | {r["cv_percent"]:.3f} | {r["rank_23"]}/{r["rank_42"]}/{r["rank_88"]} | {r["mean_rank"]} | {r["best_baseline"]} ({r["baseline_mean"]:.6f}) | {improvement} |')
lines += ['', '## 各模型的跨种子波动', '',
          '| 模型 | 平均 MAE CV % | 最大 MAE CV %（数据集） | 平均 RMSE CV % | 最大 RMSE CV %（数据集） |',
          '|---|---:|---|---:|---|']
for m in sorted(models, key=lambda m: mean(stats[d, m, 'MAE']['cv_percent'] for d in datasets)):
    cells = []
    for met in ('MAE', 'RMSE'):
        rows = [stats[d, m, met] for d in datasets]
        worst = max(rows, key=lambda r: r['cv_percent'])
        cells += [f'{mean(r["cv_percent"] for r in rows):.3f}', f'{worst["cv_percent"]:.3f} ({worst["dataset"]})']
    lines.append('| ' + ' | '.join([m] + cells) + ' |')
lines += ['', '## 各数据集的跨种子波动', '',
          '以下对 13 个可训练模型等权汇总，不含 HA。', '',
          '| 数据集 | 平均 MAE CV % | 最大 MAE CV %（模型） | 平均 RMSE CV % |',
          '|---|---:|---|---:|']
for d in datasets:
    rows = [stats[d, m, 'MAE'] for m in models if m != 'HA']
    worst = max(rows, key=lambda r: r['cv_percent'])
    lines.append(f'| {d} | {mean(r["cv_percent"] for r in rows):.3f} | {worst["cv_percent"]:.3f} ({worst["model"]}) | {mean(stats[d,m,"RMSE"]["cv_percent"] for m in models if m != "HA"):.3f} |')
lines += ['', '## 全模型均值和标准差', '',
          '每个数据集按 MAE 均值升序排列；精确方差、逐种子数值、逐种子排名及均值排名见 all_metrics.csv。']
for d in datasets:
    lines += ['', f'### {d}', '', '| 模型 | MAE | RMSE | MAPE | SMAPE | R² |', '|---|---:|---:|---:|---:|---:|']
    for m in sorted(models, key=lambda m: stats[d,m,'MAE']['mean']):
        cells = [f'{stats[d,m,met]["mean"]:.4f} ± {stats[d,m,met]["sample_std"]:.4f}' for met in METRICS]
        lines.append('| ' + ' | '.join([m] + cells) + ' |')
lines += ['', '## nyc_bike：仅 seed=88', '', '| 模型 | MAE | RMSE | MAPE | SMAPE | R² |', '|---|---:|---:|---:|---:|---:|']
for (d, m), runs in sorted(groups.items(), key=lambda item: item[1].get(88, {}).get('result', {}).get('MAE', math.inf)):
    if d == 'nyc_bike':
        lines.append('| ' + ' | '.join([m] + [f'{runs[88]["result"][met]:.4f}' for met in METRICS]) + ' |')
lines += ['', '复现：在项目根目录运行 `python docs/seed_comparison_23_42_88/analyze.py`。',
          'run_inventory.csv 保存所有实验路径及纳入/剔除原因；config_differences.csv 保存配置差异。', '']
(OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
ordered = sorted(models, key=lambda m: mean(stats[d,m,'MAE']['cv_percent'] for d in datasets))
values = [[stats[d,m,'MAE']['cv_percent'] for d in datasets] for m in ordered]
im = ax.imshow(values, cmap='YlOrRd', vmin=0)
ax.set_xticks(range(len(datasets)), datasets)
ax.set_yticks(range(len(ordered)), ordered)
for i, row in enumerate(values):
    for j, v in enumerate(row):
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='white' if v > max(map(max, values))*.6 else 'black')
ax.set_title('MAE variation across seeds 23, 42, 88\nSample standard deviation / mean (%)')
fig.colorbar(im, ax=ax, label='CV (%)')
fig.tight_layout()
fig.savefig(OUT / 'mae_cv.png', dpi=180)
plt.close(fig)
print('\n'.join(lines[:92]))
print('Inventory:', dict((s, sum(r['status'] == s for r in audit)) for s in sorted({r['status'] for r in audit})))
for met in METRICS:
    rows = [r for r in prophet if r['metric'] == met]
    print(met, 'first places:', sum(r[f'rank_{s}']==1 for r in rows for s in SEEDS), '/21;',
          'first by mean:', sum(r['mean_rank']==1 for r in rows), '/7')
