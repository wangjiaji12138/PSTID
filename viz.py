"""
PSTGN 可视化模块

将原 train.py 中所有的可视化代码抽取到本文件：

1. 预测曲线 (prediction curves)
2. 原型分布降维可视化 (prototype analysis via tSNE)
3. Temporal / Spatial query vs prototype 可视化
4. 原型使用率柱状图
5. 时间步级 Temporal Prototype 使用率折线图
6. 空间注意力可视化（每节点热力图 / 雷达图 / 聚类图）
7. Temporal Basis 可视化（热力图 / 折线 / 堆叠 / Dominant / 摘要）
8. TID / DIW 嵌入可视化
9. 工作日 vs 周末 / 高低流量天对比

使用方式：

A) 作为训练流程的一部分（由 train.py 调用）：
    from viz import run_all_visualizations
    run_all_visualizations(model, test_loader, scaler, device, args, results_dir)

B) 单独使用（加载 checkpoint 后可视化）：
    python viz.py \
        --model PSTID \
        --checkpoint results/cq/PSTID_xxx/checkpoint/best_model.pth \
        --data cq \
        --save_dir results/cq/PSTID_xxx/visualization
        
AI生成代码注意事项：
1. 所有代码都使用中文注释
2. 所有类不允许对数据格式有兼容或者是多种选择
3. 所有涉及到从train中传来的参数，如果出错（不符合格式、维度有问题等）必须返回异常
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use('Agg')  # 非交互式后端，避免阻塞
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as SkStandardScaler

# 配置中文字体支持
# matplotlib 注册的字体名为 "Noto Sans CJK JP"（TTC 索引），而非 "Noto Sans CJK SC"
# DejaVu Sans 放在最前以确保 Latin 字符优先使用带完整 Latin 覆盖的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示

# 写入 logger 'train'（使用 INFO:train: 格式前缀）
logger = logging.getLogger('train')
if not logger.handlers:
    formatter = logging.Formatter('INFO:%(name)s:%(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


# ============================================================================
# 基础数据处理
# ============================================================================

def normalize_input(X, scaler, input_dim=3):
    """Normalize input for multi-channel data.

    Args:
        X: Input tensor (B, T, N, input_dim) - first (input_dim-2) channels are data,
           last 2 channels are time features (hour, dow)
        scaler: StandardScaler for data channels
        input_dim: Number of input channels

    Returns:
        Normalized tensor with data channels normalized and time features as-is
    """
    num_data_channels = input_dim - 2  # Exclude hour and dow
    time_feats = X[..., -2:]  # Last 2 channels: hour, dow

    if isinstance(X, torch.Tensor):
        mean_t = torch.as_tensor(scaler.mean, device=X.device, dtype=X.dtype)
        std_t = torch.as_tensor(scaler.std, device=X.device, dtype=X.dtype)
        # Normalize data channels separately (each channel normalized by its own scaler)
        data_norm = (X[..., :num_data_channels] - mean_t) / std_t.clamp(min=1e-6)
    else:
        data_norm = (X[..., :num_data_channels] - scaler.mean) / max(scaler.std, 1e-6)

    return torch.cat([data_norm, time_feats], dim=-1)


# ============================================================================
# 1. 预测曲线
# ============================================================================

def plot_prediction_curves(preds, targets, save_path, num_samples=5, city_name='test'):
    """绘制预测值与真实值对比曲线
    随机选择 num_samples 个节点，在同一 output window 内展示曲线。
    preds / targets: (B, T, N)

    Y轴主刻度最小为1（可大于1），保证数据范围聚焦且可读。
    """
    B, T, N = preds.shape
    n_samples = min(num_samples, N)
    # 随机选 n_samples 个节点
    node_indices = np.random.choice(N, size=n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))

    if n_samples == 1:
        axes = [axes]

    for idx, node_idx in enumerate(node_indices):
        ax = axes[idx]
        t = np.arange(T)

        ax.plot(t, targets[0, :, node_idx], 'b-', label='Ground Truth', alpha=0.8, linewidth=1.5)
        ax.plot(t, preds[0, :, node_idx], 'r--', label='Prediction', alpha=0.8, linewidth=1.5)

        mae = np.mean(np.abs(preds[0, :, node_idx] - targets[0, :, node_idx]))
        rmse = np.sqrt(np.mean((preds[0, :, node_idx] - targets[0, :, node_idx])**2))

        ax.set_title(f'{city_name} - Node {node_idx} (MAE={mae:.4f}, RMSE={rmse:.4f})')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Y 轴刻度步长 = 1（相邻刻度差值不小于 1，避免刻度过密）：
        #   - 先按数据真实范围设置 ylim（保证曲线完整显示）
        #   - 再设置整数刻度（步长=1，覆盖 ylim），刻度值无下界限制（可以是 0、3、5…）
        y_min_data = min(preds[0, :, node_idx].min(), targets[0, :, node_idx].min())
        y_max_data = max(preds[0, :, node_idx].max(), targets[0, :, node_idx].max())
        y_range = y_max_data - y_min_data
        margin = max(y_range * 0.1, 0.3)  # 余量：10% 或至少 0.3（避免贴边）
        # ylim 严格包住数据（让 0/小值曲线也能完整显示）
        ylim_bottom = y_min_data - margin
        ylim_top = y_max_data + margin
        ax.set_ylim(ylim_bottom, ylim_top)
        # 整数刻度：步长 = 1，覆盖 ylim（刻度值本身可以是 0、1、3、5…）
        tick_start = int(np.floor(ylim_bottom))
        tick_end = int(np.ceil(ylim_top))
        ax.set_yticks(np.arange(tick_start, tick_end + 1, 1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[Saved] {save_path}")


# ============================================================================
# 2. 原型分布降维可视化（tSNE）
# ============================================================================

def plot_prototype_analysis(vis_data, save_path, city_name='test'):
    """原型分布降维可视化：Node Identity / Spatial Prototypes / Temporal Prototypes

    三个子图分别对以下内容做 tSNE 投影到2D：
      1. Node Identity（最后一层的节点身份，shape (N, hidden_dim)）
      2. 空间原型参数（n_spatial, proto_dim）
      3. 时间原型参数（n_temporal, proto_dim）

    tSNE 的优势：保留局部邻接关系，能把"几个分离的 cluster"清晰地展开；
    而 PCA 是线性投影，遇到弯曲/聚类结构时会把所有点压到中心区域。

    数据来源：统一的 ProtoVisData 对象（由 model.collect_visualization_data() 生成）

    Args:
        vis_data: ProtoVisData 对象，包含所有可视化所需的数据
        save_path: 保存路径
        city_name: 城市名称（用于标题）
    """
    assert vis_data is not None, "vis_data must not be None for plot_prototype_analysis"

    # ---- 1. 获取 Node Identity（从 vis_data）----
    raw = vis_data.identity_first_batch
    assert raw is not None, "vis_data.identity_first_batch must not be None"
    if isinstance(raw, list):
        assert len(raw) > 0, "vis_data.identity_first_batch (list) is empty"
        raw = np.concatenate(raw, axis=0)
        id_data = raw.reshape(-1, raw.shape[-1])
        n_nodes = id_data.shape[0]
    else:
        id_data = np.array(raw)
        n_nodes = id_data.shape[0]

    # ---- 2. 原型参数（从 vis_data）----
    spatial_prototypes = vis_data.spatial_prototypes  # (n_s, D)
    n_spatial = vis_data.n_spatial

    temporal_prototypes = vis_data.temporal_prototypes  # (n_t, D)
    n_temporal = vis_data.n_temporal

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---------- helper: tSNE 降维到 2D ----------
    def reduce_plot(ax, data, title, color, perplexity=30):
        """对 data (N, D) 做 tSNE 投影到 2D 后绘图

        步骤：
          1) StandardScaler 标准化（消除各维量纲差异）
          2) tSNE 投影到 2D（保留局部结构）
        """
        assert data is not None and len(data) > 0,             f"reduce_plot: data is None or empty for '{title}'" 
        n = data.shape[0]
        assert n >= 5, f"reduce_plot: tSNE requires n >= 5, got n={n} for '{title}'" 
        # perplexity 不能超过 n-1
        eff_perp = min(perplexity, max(1, n - 1))
        scaler = SkStandardScaler()
        data_std = scaler.fit_transform(data)
        tsne = TSNE(
            n_components=2,
            perplexity=eff_perp,
            random_state=42,
            init='pca',
            learning_rate='auto',
            max_iter=1000,
        )
        coords = tsne.fit_transform(data_std)

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color, alpha=0.7, s=15)
        ax.set_title(title)
        ax.set_xlabel('tSNE-1')
        ax.set_ylabel('tSNE-2')
        ax.grid(True, alpha=0.3)
        return scatter

    # 子图1：Node Identity
    reduce_plot(axes[0], id_data,
                f'Node Identity (t=0, N={n_nodes})', color='steelblue')

    # 子图2：空间原型
    reduce_plot(axes[1], spatial_prototypes,
                f'Spatial Prototypes (n={n_spatial})', color='coral')

    # 子图3：时间原型
    reduce_plot(axes[2], temporal_prototypes,
                f'Temporal Prototypes (n={n_temporal})', color='seagreen')

    fig.suptitle(f'{city_name} — Prototype Distribution (tSNE)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[Saved] {save_path}")


# ============================================================================
# 3. Query vs Prototype 可视化
# ============================================================================

def plot_temporal_query_prototype(vis_data, save_path, city_name='test', max_queries=2000):
    """Query-Prototype 可视化：左图 Temporal、右图 Spatial（同图并排）

    数据来源：统一的 ProtoVisData 对象

    Args:
        vis_data: ProtoVisData 对象，包含 temporal_query_proj, spatial_query_proj 和原型参数
        save_path: 保存路径
        city_name: 城市名称
        max_queries: 最大 query 采样数量
    """
    assert vis_data is not None, "vis_data must not be None for plot_temporal_query_prototype" 

    def build_panel(ax, query, prototypes, title_prefix, max_queries,
                    perplexity=30):
        """在指定 ax 上画 query(圆) + prototype(★)

        使用 t-SNE 代替 PCA：
          1) 先对 prototypes + queries 拼接 fit（建立联合空间）
          2) 各自 transform 回来（保持局部邻接关系）

        同时显示高维余弦相似度热力图，揭示真实匹配关系。
        """
        assert query is not None and prototypes is not None,             f"build_panel: query and prototypes must not be None (title='{title_prefix}')" 

        query_np = np.asarray(query)
        proto_np = np.asarray(prototypes)

        n_q = query_np.shape[0]
        n_p = proto_np.shape[0]

        # ---- Step 1: 高维余弦相似度（使用全部 query）----
        q_norm = query_np / (np.linalg.norm(query_np, axis=1, keepdims=True) + 1e-8)
        p_norm = proto_np / (np.linalg.norm(proto_np, axis=1, keepdims=True) + 1e-8)
        cosine_sim = q_norm @ p_norm.T
        cosine_nearest = np.argmax(cosine_sim, axis=1)
        cosine_max = np.max(cosine_sim, axis=1)

        # ---- Step 2: 下采样（按最近原型分组）----
        # 减少采样数量，让 query 和原型数量更接近，t-SNE 效果更好
        samples_per_proto = 20
        rng = np.random.default_rng(42)
        q_indices = []
        used_pids = []  # 记录实际有 query 指向的原型（用于后续 prototype 着色）
        collapsed_pids = []  # 记录坍缩的原型 id，最后聚合输出
        for pid in range(n_p):
            mask = (cosine_nearest == pid)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                # prototype collapse: 该原型在所有 query 中没有最近邻
                # 暂存，最后聚合输出（避免每原型一行 warning 淹没日志）
                collapsed_pids.append(pid)
                continue
            used_pids.append(pid)
            n_sel = min(samples_per_proto, len(indices))
            q_indices.extend(rng.choice(indices, n_sel, replace=False))
        if collapsed_pids:
            pass  # Prototype collapse warning (optional debug info)

        assert len(q_indices) > 0, \
            f"build_panel({title_prefix}): no query samples available — check input data"

        q_indices = np.array(q_indices)
        n_q_sampled = len(q_indices)
        q_data = query_np[q_indices]

        # ---- Step 3: t-SNE fit + transform（联合空间，更好地保留局部结构）----
        # 分别标准化 query 和 prototype，避免数值范围差异导致分布不均
        scaler_q = SkStandardScaler()
        scaler_p = SkStandardScaler()
        q_std = scaler_q.fit_transform(q_data)
        p_std = scaler_p.fit_transform(proto_np)
        combined_std = np.vstack([p_std, q_std])
        n_proto = len(proto_np)

        # t-SNE perplexity 需要在 sqrt(n) 左右，且小于 n-1
        perp_range = min(n_q_sampled + n_proto - 1, 50)
        perplexity = max(5, min(perp_range, (n_q_sampled + n_proto) // 4))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        combined_2d = tsne.fit_transform(combined_std)

        p_2d = combined_2d[:n_proto]
        q_2d = combined_2d[n_proto:]

        # ---- Step 4: 着色 ----
        if n_p <= 12:
            cmap = plt.colormaps['Set3']
            p_colors = [cmap(i % cmap.N) for i in range(n_p)]
        elif n_p <= 20:
            cmap = plt.colormaps['Set2']
            p_colors = [cmap(i % cmap.N) for i in range(n_p)]
        else:
            try:
                import seaborn as sns
                p_colors = sns.color_palette('husl', n_colors=n_p)
            except ImportError:
                p_colors = []
                for i in range(n_p):
                    hue = (0.55 + i / n_p) % 1.0
                    p_colors.append(plt.matplotlib.colors.hsv_to_rgb((hue, 0.55, 0.85)))

        q_colors = [p_colors[cosine_nearest[q_indices[i]]] for i in range(n_q_sampled)]

        # 用余弦相似度调节透明度：相似度越高越不透明
        q_alphas = [0.3 + 0.7 * cosine_max[q_indices[i]] for i in range(n_q_sampled)]

        ax.scatter(q_2d[:, 0], q_2d[:, 1],
                  c=q_colors, alpha=0.55, s=60,
                  edgecolors='none',
                  label=f'{title_prefix}-Query', zorder=1)

        # 连线
        for i in range(n_q_sampled):
            pid = cosine_nearest[q_indices[i]]
            ax.plot([q_2d[i, 0], p_2d[pid, 0]],
                    [q_2d[i, 1], p_2d[pid, 1]],
                    c=p_colors[pid], alpha=0.15, linewidth=0.6, zorder=0)

        # prototype: ★
        ax.scatter(p_2d[:, 0], p_2d[:, 1],
                   c=p_colors, marker='*', s=500,
                   edgecolors='black', linewidths=0.8,
                   label=f'{title_prefix}-Proto', zorder=5)

        title_suffix = '' if len(used_pids) == n_proto else f' [{len(used_pids)}/{n_proto} used]'
        ax.set_title(f'{title_prefix} (n_q={n_q_sampled}, n_p={n_proto}){title_suffix}')
        ax.set_xlabel('t-SNE-1')
        ax.set_ylabel('t-SNE-2')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        return None

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Temporal
    build_panel(axes[0], vis_data.temporal_query_proj, vis_data.temporal_prototypes,
                title_prefix='Temporal',
                max_queries=max_queries)

    # Spatial
    build_panel(axes[1], vis_data.spatial_query_proj, vis_data.spatial_prototypes,
                title_prefix='Spatial',
                max_queries=max_queries)

    layer_label = f'layer {vis_data.layer_idx + 1}'
    fig.suptitle(f'{city_name} — Query vs Prototype (tSNE, {layer_label})', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[Saved] {save_path}")


# ============================================================================
# 4. 原型使用率柱状图
# ============================================================================

def plot_temporal_per_timestep_usage(vis_data, save_path, city_name='test'):
    """时间步级 Temporal Prototype 使用率折线图

    展示不同 time_of_day 对应的 16 个 temporal prototype 使用率变化，
    每个 prototype 一条折线，横轴是实际时间（小时），纵轴是使用率。
    数据按样本的最后一个时间步对应的 time_of_day 进行键值匹配聚合。

    数据来源：统一的 ProtoVisData 对象

    Args:
        vis_data: ProtoVisData 对象，包含 temporal_by_time_of_day dict
        save_path: 保存路径
        city_name: 城市名称
    """
    assert vis_data is not None, "vis_data must not be None for plot_temporal_per_timestep_usage"

    assert hasattr(vis_data, 'temporal_by_time_of_day'),         "vis_data must have 'temporal_by_time_of_day' attribute"
    temporal_by_tod = vis_data.temporal_by_time_of_day
    assert temporal_by_tod is not None and len(temporal_by_tod) > 0,         "vis_data.temporal_by_time_of_day must not be None or empty" 

    # 获取时间槽数量和原型数量
    time_of_day_size = vis_data.time_of_day_size
    n_temporal = list(temporal_by_tod.values())[0].shape[0]

    # 排序 time_of_day 索引
    sorted_tods = sorted(temporal_by_tod.keys())

    # 创建时间标签和 x 轴（只显示实际存在的时间槽）
    x_positions = list(sorted_tods)  # 实际存在的 time slot 位置
    x_labels = []

    for tod in sorted_tods:
        # 将 tod 索引转换为小时
        hour = tod * 24.0 / time_of_day_size
        hour_int = int(hour) % 24
        minute = int((hour - hour_int) * 60)
        if minute == 0:
            x_labels.append(f'{hour_int:02d}:00')
        else:
            x_labels.append(f'{hour_int:02d}:{minute:02d}')

    # 构建完整的时间序列（使用实际的 time_of_day_size）
    full_temporal = np.zeros((time_of_day_size, n_temporal))
    for tod, usage in temporal_by_tod.items():
        full_temporal[tod] = usage

    # ---- 创建可视化 ----
    fig, ax = plt.subplots(figsize=(16, 7))

    # 使用颜色映射
    try:
        import seaborn as sns
        colors = sns.color_palette('husl', n_colors=n_temporal)
    except ImportError:
        # seaborn is optional; fall back to matplotlib colormap
        colors = [plt.colormaps['tab20'](i / max(n_temporal - 1, 1)) for i in range(n_temporal)]

    # ---- 折线图: 每个 prototype ----
    max_values = []  # 每个 prototype 的最大值
    max_indices = []  # 最大值对应的时间槽索引

    for proto_idx in range(n_temporal):
        # 只取 x_positions 对应的索引
        usage_values = full_temporal[x_positions, proto_idx]

        # 绘制折线
        ax.plot(x_positions, usage_values,
                color=colors[proto_idx],
                linewidth=1.5,
                marker='o', markersize=4,
                alpha=0.7,
                label=f'Proto {proto_idx}')

        # 找到最大值
        max_val = np.max(usage_values)
        max_idx_in_array = np.argmax(usage_values)
        max_x_pos = x_positions[max_idx_in_array]  # 实际的 x 坐标
        max_values.append(max_val)
        max_indices.append(max_x_pos)

        # 在最大值处标注
        ax.scatter([max_x_pos], [max_val], color=colors[proto_idx], s=80, zorder=5)
        # 找到对应的 x_labels
        if max_x_pos in sorted_tods:
            max_label_idx = sorted_tods.index(max_x_pos)
            max_time_label = x_labels[max_label_idx]
            ax.annotate(f'P{proto_idx}:{max_time_label}',
                        xy=(max_x_pos, max_val),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=colors[proto_idx],
                        fontweight='bold')
        else:
            ax.annotate(f'P{proto_idx}:{max_val:.2f}',
                        xy=(max_x_pos, max_val),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=colors[proto_idx],
                        fontweight='bold')

    # 找出总体最大值的 prototype 并高亮
    overall_max_proto = np.argmax(max_values)
    overall_max_val = max_values[overall_max_proto]
    overall_max_x_pos = max_indices[overall_max_proto]

    # 重新绘制最大 prototype 的高亮线
    ax.plot(x_positions, full_temporal[x_positions, overall_max_proto],
            color=colors[overall_max_proto],
            linewidth=3, alpha=0.9,
            label=f'★ Proto {overall_max_proto} (max={overall_max_val:.3f})')
    ax.scatter([overall_max_x_pos], [overall_max_val],
               color=colors[overall_max_proto], s=120, zorder=6, edgecolor='black', linewidth=2)

    ax.set_xlabel('Time of Day (Hour)', fontsize=12)
    ax.set_ylabel('Usage Rate', fontsize=12)
    ax.set_title(f'{city_name} — Temporal Prototype Usage by Time-of-Day (n_temporal={n_temporal})',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              ncol=1, fontsize=9, framealpha=0.9)

    # 添加统计信息
    if overall_max_x_pos in sorted_tods:
        max_label_idx = sorted_tods.index(overall_max_x_pos)
        max_time_label = x_labels[max_label_idx]
        stats_text = f"{len(temporal_by_tod)} time slots\nn_temporal={n_temporal}\nMax: Proto {overall_max_proto}\nPeak: {max_time_label}"
    else:
        stats_text = f"{len(temporal_by_tod)} time slots\nn_temporal={n_temporal}\nMax: Proto {overall_max_proto}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[Saved] {save_path}")


# ============================================================================
# 6. 空间注意力可视化
# ============================================================================

def plot_spatial_attention_per_node(spatial_att_per_node, n_spatial, output_dir, dataset_info, args):
    """绘制每个节点对空间原型的注意力喜好热力图"""
    import matplotlib.gridspec as gridspec
    N, n_sp = spatial_att_per_node.shape

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 按主导原型排序
    dominant_proto = spatial_att_per_node.argmax(axis=1)
    sort_idx = np.argsort(dominant_proto)
    att_sorted = spatial_att_per_node[sort_idx]

    # 热力图
    im = axes[0].imshow(att_sorted.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xlabel('Node ID (sorted by dominant prototype)', fontsize=12)
    axes[0].set_ylabel('Spatial Prototype Index', fontsize=12)
    axes[0].set_title(f'{args.data} — Spatial Attention per Node\n(N={N}, n_spatial={n_spatial})', fontsize=11)
    plt.colorbar(im, ax=axes[0], label='Attention Weight')

    # 每个原型的节点数统计
    proto_counts = np.bincount(dominant_proto, minlength=n_spatial)
    proto_percentages = proto_counts / N * 100

    colors = plt.cm.tab20(np.linspace(0, 1, n_spatial))
    bars = axes[1].bar(range(n_spatial), proto_counts, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Spatial Prototype Index', fontsize=12)
    axes[1].set_ylabel('Number of Nodes', fontsize=12)
    axes[1].set_title(f'{args.data} — Dominant Prototype Distribution', fontsize=11)
    axes[1].set_xticks(range(n_spatial))

    for bar, pct in zip(bars, proto_percentages):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{args.data} — Spatial Prototype Preference', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / 'spatial_attention_per_node.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[Saved] {save_path}")


def plot_spatial_attention_radar(spatial_att_per_node, n_spatial, output_dir, dataset_info, args):
    """绘制雷达图展示代表性节点的注意力喜好"""
    N, n_sp = spatial_att_per_node.shape

    # 随机选择 10 个代表性节点
    np.random.seed(42)
    selected_nodes = np.random.choice(N, size=min(10, N), replace=False)
    selected_labels = [f'Node {n}' for n in selected_nodes]

    selected_att = spatial_att_per_node[selected_nodes]

    # 雷达图
    angles = np.linspace(0, 2 * np.pi, n_spatial, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    max_val = max(0.15, selected_att.max() * 1.2)

    for i, (node_idx, label) in enumerate(zip(selected_nodes, selected_labels)):
        values = selected_att[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i], alpha=0.8)
        ax.fill(angles, values, alpha=0.05, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'Proto {i}' for i in range(n_spatial)], fontsize=10)
    ax.set_ylim(0, max_val)
    ax.set_yticks(np.linspace(0, max_val, 5))
    ax.set_yticklabels([f'{v:.3f}' for v in np.linspace(0, max_val, 5)], fontsize=8)
    ax.set_title(f'{args.data} — Spatial Attention Radar\n'
                 f'(Random 10 nodes)', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    save_path = output_dir / 'spatial_attention_radar.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[Saved] {save_path}")


def plot_spatial_attention_cluster(spatial_att_per_node, n_spatial, output_dir, dataset_info, args):
    """使用聚类分析可视化节点的空间注意力模式

    左图：基于空间注意力的节点聚类（PCA降维）
    右图：同一簇节点在地图上的地理分布，验证注意力聚类是否与地理位置相关
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    N, n_sp = spatial_att_per_node.shape
    n_clusters = min(n_spatial, 8)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(spatial_att_per_node)
    cluster_centers = kmeans.cluster_centers_

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(spatial_att_per_node)
    center_coords = pca.transform(cluster_centers)

    # 读取节点地理坐标
    city = args.data
    metadata_path = Path(args.data_dir) / city / 'metadata.json'
    assert metadata_path.exists(), \
        f"plot_spatial_attention_cluster: metadata.json not found at {metadata_path}"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    assert 'node_positions' in metadata, \
        f"plot_spatial_attention_cluster: 'node_positions' missing from {metadata_path}"
    assert 'grid_boundaries' in metadata, \
        f"plot_spatial_attention_cluster: 'grid_boundaries' missing from {metadata_path}"
    node_positions = metadata['node_positions']
    grid_boundaries = metadata['grid_boundaries']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # ========== 左图：基于注意力的聚类（PCA降维） ==========
    for cluster_idx in range(n_clusters):
        mask = cluster_labels == cluster_idx
        axes[0].scatter(coords[mask, 0], coords[mask, 1], c=[colors[cluster_idx]], alpha=0.6, s=30,
                       label=f'Cluster {cluster_idx} (n={mask.sum()})')

    for cluster_idx in range(n_clusters):
        axes[0].scatter(center_coords[cluster_idx, 0], center_coords[cluster_idx, 1],
                       c='black', marker='X', s=200, edgecolors='white', linewidths=2, zorder=10)

    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    axes[0].set_title(f'{args.data} — Node Clustering by Spatial Attention (K-Means, n_clusters={n_clusters})', fontsize=11)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # ========== 右图：节点的地理分布（按聚类着色） ==========
    # 提取经纬度
    lngs = [p['lng_center'] for p in node_positions]
    lats = [p['lat_center'] for p in node_positions]

    # 绘制每个节点，颜色根据簇标签确定
    for node_idx, pos in enumerate(node_positions):
        if node_idx < len(cluster_labels):
            cluster_idx = cluster_labels[node_idx]
            axes[1].scatter(pos['lng_center'], pos['lat_center'],
                           c=[colors[cluster_idx]], s=80, alpha=0.7,
                           edgecolors='white', linewidths=0.5)

    # 添加簇图例
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=colors[i], markersize=10,
                                  label=f'Cluster {i}') for i in range(n_clusters)]
    axes[1].legend(handles=legend_handles, loc='upper right', fontsize=9)

    axes[1].set_xlabel('Longitude', fontsize=12)
    axes[1].set_ylabel('Latitude', fontsize=12)
    axes[1].set_title(f'{args.data} — Geographic Distribution by Cluster', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # 绘制城市边界
    assert 'lng_min' in grid_boundaries and 'lng_max' in grid_boundaries, \
        "grid_boundaries must contain 'lng_min' and 'lng_max'"
    assert 'lat_min' in grid_boundaries and 'lat_max' in grid_boundaries, \
        "grid_boundaries must contain 'lat_min' and 'lat_max'"
    min_lng = grid_boundaries['lng_min']
    max_lng = grid_boundaries['lng_max']
    min_lat = grid_boundaries['lat_min']
    max_lat = grid_boundaries['lat_max']
    axes[1].set_xlim(min_lng - 0.01, max_lng + 0.01)
    axes[1].set_ylim(min_lat - 0.01, max_lat + 0.01)

    fig.suptitle(f'{args.data} — Spatial Attention Cluster Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / 'spatial_attention_cluster.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[Saved] {save_path}")


def collect_spatial_attention_per_node(model, test_loader, device, args):
    """收集测试集上每个节点的空间注意力权重"""
    model.eval()

    n_spatial = model.num_spatial_prototypes
    num_nodes = model.num_nodes

    # 初始化累积器（因为 collect_visualization_data 可能已清理）
    model._proto_vis_accumulator = {
        'tod_last': [],
        'temporal_att': [],
    }
    assert model.proto_module is not None, \
        "collect_spatial_attention_per_node: model.proto_module must not be None"
    model.proto_module._proto_vis_accumulator = {
        'tod_last': [],
        'temporal_att': [],
    }

    accum_att = np.zeros((num_nodes, n_spatial))
    accum_counts = np.zeros(num_nodes)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            X = batch['X'].to(device)
            # 构造模型需要的输入格式
            model_input = {'X': X}
            model.forward(model_input, collect_cache=True)

            usages = model.proto_module._last_proto_usage
            assert usages is not None and len(usages) > 0, \
                "collect_spatial_attention_per_node: model.proto_module._last_proto_usage must not be None or empty"
            u = usages[0]
            assert 'spatial_per_node' in u, \
                "collect_spatial_attention_per_node: 'spatial_per_node' not found in _last_proto_usage[0]"

            batch_att = u['spatial_per_node']
            accum_att += batch_att
            accum_counts += 1

            if batch_idx >= 50:
                break

    if accum_counts.sum() > 0:
        spatial_att_per_node = accum_att / accum_counts[:, np.newaxis]
    else:
        spatial_att_per_node = accum_att

    return spatial_att_per_node


def run_visualization(model, test_loader, scaler, device, args, save_dir):
    """在测试集上运行推理，收集预测结果和原型信息，生成可视化文件"""
    model.eval()

    all_preds = []
    all_targets = []

    # 判断是否为 PSTID 模型且启用了原型
    is_proto_model = args.model.upper() == 'PSTID' and args.use_proto

    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y_raw = batch['y'].to(device)

            X_norm = normalize_input(X, scaler, args.input_dim)

            # PSTID 模型在需要可视化时启用缓存收集
            if is_proto_model:
                # 使用 collect_cache=True 来收集可视化数据
                output = model({'X': X_norm}, collect_cache=True)
            else:
                output = model({'X': X_norm})

            y_pred_norm = output.mean(dim=-1, keepdim=True)
            y_pred_raw = scaler.inverse_transform(y_pred_norm)
            y_target_avg = y_raw.mean(dim=-1, keepdim=True)

            all_preds.append(y_pred_raw)
            all_targets.append(y_target_avg)

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()

    vis_dir = Path(save_dir) / 'visualization'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. 预测曲线对比（随机选5节点，同一output window）
    # all_preds / all_targets: (B, T, N, 1) → (B, T, N)
    plot_prediction_curves(
        all_preds[..., 0], all_targets[..., 0],
        vis_dir / 'prediction_curves.png',
        num_samples=5,
        city_name=args.data
    )

    # 2. 原型分布降维可视化（identity / spatial / temporal）
    vis_data = None
    if args.use_proto and hasattr(model, 'collect_visualization_data'):
        vis_data = model.collect_visualization_data(layer_idx=0)
        assert vis_data is not None,             "run_visualization: model.collect_visualization_data() returned None"
        plot_prototype_analysis(vis_data, vis_dir / 'prototype_analysis.png', city_name=args.data)

    # 2b. 时间码本：query (圆点) vs 原型 (星形)
    if args.use_proto and args.use_temporal and hasattr(model, 'collect_visualization_data'):
        vis_data = model.collect_visualization_data(layer_idx=0)
        assert vis_data is not None,             "run_visualization: model.collect_visualization_data() returned None (temporal)"
        plot_temporal_query_prototype(vis_data, vis_dir / 'temporal_query_prototype.png',
                                      city_name=args.data, max_queries=2000)

    # 3. 时间步级 Temporal Prototype 使用率折线图
    if args.use_proto and args.use_temporal and hasattr(model, 'collect_visualization_data'):
        vis_data = model.collect_visualization_data(layer_idx=0)
        assert vis_data is not None,             "run_visualization: model.collect_visualization_data() returned None (per-timestep)"
        plot_temporal_per_timestep_usage(vis_data, vis_dir / 'temporal_per_timestep_usage.png',
                                        city_name=args.data)

    # 保存预测数据（保存所有节点）
    np.save(vis_dir / 'predictions.npy', all_preds[..., 0])
    np.save(vis_dir / 'targets.npy', all_targets[..., 0])


# ============================================================================
# 7. Temporal Basis 可视化
# ============================================================================

def run_temporal_basis_visualization(model, test_loader, scaler, device, args, save_dir):
    """运行 Temporal Basis 可视化

    集成自 visualize_temporal_basis.py 的功能
    """
    output_dir = Path(save_dir) / 'visualization'
    output_dir.mkdir(parents=True, exist_ok=True)


    # ========== 1. 分析数据集时间范围 ==========
    dataset_info = analyze_time_slots_for_visualization(test_loader)

    # ========== 2. 收集 Temporal Basis 权重 ==========
    temporal_weights = collect_temporal_basis_weights(
        model, test_loader, scaler, device, args, dataset_info
    )

    # 检查是否有 temporal basis 数据
    has_temporal_data = np.sum(temporal_weights['counts_per_timestep'] > 0) > 0
    assert has_temporal_data,         "run_temporal_basis_visualization: no temporal basis data (all counts_per_timestep == 0)"

    if has_temporal_data:
        # 保存权重文件
        save_temporal_weights_json(temporal_weights, output_dir, dataset_info, args)
        save_temporal_weights_csv(temporal_weights, output_dir, dataset_info, args)

        # 生成 Temporal Basis 可视化
        plot_temporal_basis_line(temporal_weights, output_dir, dataset_info, args)

    # ========== 3. 收集并可视化每个节点的空间注意力喜好 ==========
    if model.use_spatio and model.num_spatial_prototypes > 0:
        spatial_att_per_node = collect_spatial_attention_per_node(
            model, test_loader, device, args
        )

        # 生成空间注意力可视化
        plot_spatial_attention_per_node(spatial_att_per_node, model.num_spatial_prototypes, output_dir, dataset_info, args)
        plot_spatial_attention_radar(spatial_att_per_node, model.num_spatial_prototypes, output_dir, dataset_info, args)
        plot_spatial_attention_cluster(spatial_att_per_node, model.num_spatial_prototypes, output_dir, dataset_info, args)

        # 保存每个节点的注意力权重到 JSON
        spatial_att_dict = {
            'num_nodes': int(spatial_att_per_node.shape[0]),
            'num_spatial_prototypes': int(spatial_att_per_node.shape[1]),
            'per_node_attention': {
                f'node_{i}': spatial_att_per_node[i].tolist()
                for i in range(spatial_att_per_node.shape[0])
            },
            'dominant_prototype': spatial_att_per_node.argmax(axis=1).tolist(),
            'mean_attention_per_proto': spatial_att_per_node.mean(axis=0).tolist(),
        }
        with open(output_dir / 'spatial_attention_per_node.json', 'w') as f:
            json.dump(spatial_att_dict, f, indent=2)
    else:
        pass  # Spatial attention visualization skipped (spatio not enabled)

    return output_dir


def analyze_time_slots_for_visualization(test_loader):
    """分析测试集的时间范围"""
    all_time_of_day = []
    all_day_of_week = []

    for batch in test_loader:
        X = batch['X']
        time_of_day = X[:, -1, 0, 1].numpy()  # 最后时间步的 time_of_day
        day_of_week = X[:, -1, 0, 2].numpy()
        all_time_of_day.append(time_of_day)
        all_day_of_week.append(day_of_week)

    all_time_of_day = np.concatenate(all_time_of_day)
    all_day_of_week = np.concatenate(all_day_of_week)

    hours = all_time_of_day * 24
    unique_hours_rounded = np.unique(np.round(hours).astype(int))

    # 修复: hour=23.5 会被 round() 四舍五入为 24，
    # 但这不意味着数据覆盖了真正的 24:00 (00:00)
    # 使用 clip 限制范围: 0-23 (23.999...h 对应 23:59)
    unique_hours_rounded = np.clip(unique_hours_rounded, 0, 23)

    # 更精确的检测: 覆盖至少 20 个小时认为是 24h 数据
    hour_range = unique_hours_rounded.max() - unique_hours_rounded.min()
    is_8_20 = (unique_hours_rounded.min() >= 7) and (unique_hours_rounded.max() <= 21)
    is_24h = (hour_range >= 20) or (unique_hours_rounded.min() < 6) or (unique_hours_rounded.max() >= 22)

    if is_8_20 and not is_24h:
        dataset_type = '8:00-20:00'
        actual_time_slots = 24  # 8:00-20:00 = 12h, 30min intervals = 24 slots
        time_of_day_start = 8.0 / 24
        time_of_day_end = 20.0 / 24
    elif is_24h:
        dataset_type = '24h'
        actual_time_slots = 48  # 24h, 30min intervals = 48 slots
        time_of_day_start = 0.0
        time_of_day_end = 1.0
    else:
        dataset_type = 'mixed'
        actual_time_slots = 24
        time_of_day_start = 0.0
        time_of_day_end = 1.0

    return {
        'samples': len(all_time_of_day),
        'time_slots': actual_time_slots,
        'min_hour': int(unique_hours_rounded.min()),
        'max_hour': int(unique_hours_rounded.max()),
        'type': dataset_type,
        'time_of_day_size': actual_time_slots,
        'time_of_day_start': time_of_day_start,
        'time_of_day_end': time_of_day_end,
    }


def collect_temporal_basis_weights(model, test_loader, scaler, device, args, dataset_info):
    """收集测试集上的 temporal prototype 权重

    同时分别收集工作日和周末的数据用于对比分析。
    """
    model.eval()

    all_temporal_weights = []
    all_time_of_day = []
    all_day_of_week = []

    dataset_type = dataset_info['type']
    actual_time_slots = dataset_info['time_slots']

    assert hasattr(model, 'proto_module') and model.proto_module is not None,         "collect_temporal_basis_weights: model.proto_module must not be None"
    is_proto_model = True

    temporal_basis = model.proto_module.temporal_basis
    assert temporal_basis is not None, \
        "collect_temporal_basis_weights: model.proto_module.temporal_basis must not be None"

    n_temporal = model.num_temporal_prototypes
    assert n_temporal > 0, f"model.num_temporal_prototypes must be > 0, got {n_temporal}" 

    # ===== 分别收集工作日和周末的数据 =====
    weekday_temporal_weights = []
    weekday_time_of_day = []
    weekend_temporal_weights = []
    weekend_time_of_day = []
    weekday_flow = []
    weekend_flow = []

    # ===== 计数：哪些 batch 全是 weekend / 全是 weekday，最后聚合输出 =====
    _weekday_skipped_batches = 0
    _weekend_skipped_batches = 0

    # 用于找流量差异最大的两天 - 使用 day_of_week 来分组
    all_day_flows = {}  # {day_key: [flow_values]} 按天聚合流量
    all_day_tod = {}   # {day_key: [time_of_day_values]}
    all_day_weights = {}  # {day_key: [temporal_weights]}

    # 用于跟踪每个 dow 的当前 day_key 和 time_of_day 索引
    # 只有当 time_of_day 重置（变小）时才认为进入新的一天
    dow_current_day = {}  # {dow: current_day_key}
    dow_last_tod_idx = {}  # {dow: last_time_of_day_index}

    # 用于显示的星期几名称
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            B, T, N, _ = X.shape

            X_norm = normalize_input(X, scaler, args.input_dim)

            # 记录时间信息
            time_of_day = X[:, -1, 0, 1].cpu().numpy()
            day_of_week = X[:, -1, 0, 2].cpu().numpy()
            all_time_of_day.append(time_of_day)
            all_day_of_week.append(day_of_week)

            # 记录流量数据（取所有节点流量的均值）
            flow_per_sample = X[:, -1, :, 0].mean(dim=1).cpu().numpy()  # (B,)

            # 计算 day_of_week 的原始值 (0=Mon, 6=Sun)
            dow_values = (day_of_week * 7).astype(int).clip(0, 6)

            # 计算 time_of_day 的索引 (0-47 for 48 slots, 0-23 for 24 slots)
            dataset_type = dataset_info['type']
            actual_time_slots = dataset_info['time_slots']

            if dataset_type == '8:00-20:00':
                time_of_day_start = 8.0 / 24
                time_of_day_end = 20.0 / 24
                time_of_day_range = time_of_day_end - time_of_day_start
                tod_normalized = (time_of_day - time_of_day_start) / time_of_day_range
                tod_indices = (tod_normalized * (actual_time_slots - 1)).astype(int).clip(0, actual_time_slots - 1)
            else:
                tod_indices = (time_of_day * (actual_time_slots - 1)).astype(int).clip(0, actual_time_slots - 1)

            # 为每个样本分配 day_key 并收集数据
            # day_key 只有在 time_of_day 重置（变小）时才更新
            sample_day_keys = []  # 存储每个样本的 day_key
            for i in range(B):
                dow = dow_values[i]
                tod_idx = tod_indices[i]

                # 初始化 dow 的跟踪
                if dow not in dow_current_day:
                    dow_current_day[dow] = f"{dow}_1"
                    dow_last_tod_idx[dow] = tod_idx
                    day_key = dow_current_day[dow]
                # 检测 time_of_day 重置（新的一天）
                elif tod_idx < dow_last_tod_idx[dow]:
                    # 提取当前计数器并递增
                    current_counter = int(dow_current_day[dow].split('_')[1])
                    dow_current_day[dow] = f"{dow}_{current_counter + 1}"
                    day_key = dow_current_day[dow]
                else:
                    day_key = dow_current_day[dow]

                dow_last_tod_idx[dow] = tod_idx
                sample_day_keys.append(day_key)

                if day_key not in all_day_flows:
                    all_day_flows[day_key] = []
                    all_day_tod[day_key] = []
                    all_day_weights[day_key] = []
                all_day_flows[day_key].append(flow_per_sample[i])
                all_day_tod[day_key].append(time_of_day[i])

            # ===== 区分工作日/周末 =====
            dow_indices = (day_of_week * 7).astype(int).clip(0, 6)  # 0=Mon, 6=Sun
            is_weekend = dow_indices >= 5  # True for Sat/Sun

            # 计算 temporal attention weights
            temp = temporal_basis.temp.item()

            # 获取嵌入
            _, _, time_in_day_emb, day_in_week_emb, _ = model._get_embeddings({'X': X})
            assert time_in_day_emb is not None, "collect_temporal_basis_weights: time_in_day_emb is None"
            assert day_in_week_emb is not None, "collect_temporal_basis_weights: day_in_week_emb is None"

            # time_in_day_emb: (B, T, N, tid_dim) - 最后时间步
            tid_last = time_in_day_emb[:, -1, :, :]  # (B, N, tid_dim)

            # day_in_week_emb: (B, T, N, diw_dim) - 最后时间步
            diw_last = day_in_week_emb[:, -1, :, :]  # (B, N, diw_dim)

            # 获取 time_series_emb
            input_data = X
            batch_size, T, num_nodes, _ = input_data.shape

            time_series = input_data[..., :1]
            time_series = time_series.transpose(1, 2).contiguous()
            time_series = time_series.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            ts_emb = model.time_series_emb_layer(time_series).expand(-1, -1, -1, T).permute(0, 3, 2, 1)
            ts_last = ts_emb[:, -1, :, :]  # (B, N, ts_dim)

            # 拼接并计算 attention
            x_alloc = torch.cat([ts_last, tid_last, diw_last], dim=-1)  # (B, N, in_dim)
            x_alloc_flat = x_alloc.reshape(batch_size * num_nodes, -1)  # (B*N, in_dim)
            base_attn = temporal_basis.allocator(x_alloc_flat)  # (B*N, n_temporal)
            base_attn = F.softmax(base_attn / temp, dim=-1)

            # 对 N 求均值
            att_per_sample = base_attn.reshape(batch_size, num_nodes, -1).mean(dim=1).cpu().numpy()  # (B, n_temporal)
            all_temporal_weights.append(att_per_sample)

            # ===== 收集每天的数据用于对比 =====
            for i in range(batch_size):
                day_key = sample_day_keys[i]
                all_day_weights.setdefault(day_key, []).append(att_per_sample[i])
                _flow_val = flow_per_sample[i]
                if isinstance(_flow_val, torch.Tensor):
                    _flow_val = _flow_val.cpu().numpy()
                all_day_flows.setdefault(day_key, []).append(_flow_val)

            # ===== 分别收集工作日和周末 =====
            weekday_mask = ~is_weekend
            weekend_mask = is_weekend

            n_weekday, n_weekend = int(weekday_mask.sum()), int(weekend_mask.sum())
            if n_weekday == 0:
                # 暂存，最后聚合输出（避免每 batch 一行 warning 淹没日志）
                _weekday_skipped_batches += 1
            else:
                weekday_temporal_weights.append(att_per_sample[weekday_mask])
                weekday_time_of_day.append(time_of_day[weekday_mask])
                weekday_flow.append(flow_per_sample[weekday_mask])

            if n_weekend == 0:
                # 暂存，最后聚合输出（避免每 batch 一行 warning 淹没日志）
                _weekend_skipped_batches += 1
            else:
                weekend_temporal_weights.append(att_per_sample[weekend_mask])
                weekend_time_of_day.append(time_of_day[weekend_mask])
                weekend_flow.append(flow_per_sample[weekend_mask])

    assert len(all_temporal_weights) > 0,         "collect_temporal_basis_weights: no temporal weights collected — check temporal_basis and model._get_embeddings()" 

    all_temporal_weights = np.concatenate(all_temporal_weights, axis=0)
    all_time_of_day = np.concatenate(all_time_of_day, axis=0)
    all_day_of_week = np.concatenate(all_day_of_week, axis=0)

# ===== 合并工作日和周末数据 =====
    assert len(weekday_temporal_weights) > 0, \
        "collect_temporal_basis_weights: no weekday temporal weights collected"
    all_weekday_weights = np.concatenate(weekday_temporal_weights, axis=0)
    all_weekday_tod = np.concatenate(weekday_time_of_day, axis=0)
    assert len(weekday_flow) > 0, \
        "collect_temporal_basis_weights: no weekday flow data collected"
    all_weekday_flow = np.concatenate(weekday_flow, axis=0)

    if len(weekend_temporal_weights) == 0:
        pass  # No weekend samples warning (optional debug info)
        all_weekend_weights = all_weekday_weights
        all_weekend_tod = all_weekday_tod
        all_weekend_flow = all_weekday_flow
    else:
        all_weekend_weights = np.concatenate(weekend_temporal_weights, axis=0)
        all_weekend_tod = np.concatenate(weekend_time_of_day, axis=0)
        all_weekend_flow = np.concatenate(weekend_flow, axis=0)

    # 聚合到每个时间槽
    per_timestep = np.zeros((actual_time_slots, n_temporal))
    counts_per_timestep = np.zeros(actual_time_slots)

    per_timestep_weekday = np.zeros((actual_time_slots, n_temporal))
    per_timestep_weekend = np.zeros((actual_time_slots, n_temporal))
    counts_weekday = np.zeros(actual_time_slots)
    counts_weekend = np.zeros(actual_time_slots)
    flow_per_slot_weekday = np.zeros(actual_time_slots)
    flow_per_slot_weekend = np.zeros(actual_time_slots)

    # 计算每个时间槽的平均权重
    tod_indices_all = (all_time_of_day * (actual_time_slots - 1)).astype(int).clip(0, actual_time_slots - 1)

    for i in range(len(tod_indices_all)):
        idx = tod_indices_all[i]
        per_timestep[idx] += all_temporal_weights[i]
        counts_per_timestep[idx] += 1

    # weekday/weekend
    assert len(all_weekday_weights) > 0, \
        "collect_temporal_basis_weights: all_weekday_weights is empty"
    tod_indices_weekday = (all_weekday_tod * (actual_time_slots - 1)).astype(int).clip(0, actual_time_slots - 1)
    for i in range(len(tod_indices_weekday)):
        idx = tod_indices_weekday[i]
        per_timestep_weekday[idx] += all_weekday_weights[i]
        counts_weekday[idx] += 1
        flow_per_slot_weekday[idx] += all_weekday_flow[i]

    assert len(all_weekend_weights) > 0, \
        "collect_temporal_basis_weights: all_weekend_weights is empty"
    tod_indices_weekend = (all_weekend_tod * (actual_time_slots - 1)).astype(int).clip(0, actual_time_slots - 1)
    for i in range(len(tod_indices_weekend)):
        idx = tod_indices_weekend[i]
        per_timestep_weekend[idx] += all_weekend_weights[i]
        counts_weekend[idx] += 1
        flow_per_slot_weekend[idx] += all_weekend_flow[i]

    # 求均值
    for i in range(actual_time_slots):
        if counts_per_timestep[i] > 0:
            per_timestep[i] /= counts_per_timestep[i]
        if counts_weekday[i] > 0:
            per_timestep_weekday[i] /= counts_weekday[i]
            flow_per_slot_weekday[i] /= counts_weekday[i]
        if counts_weekend[i] > 0:
            per_timestep_weekend[i] /= counts_weekend[i]
            flow_per_slot_weekend[i] /= counts_weekend[i]

    # ===== 计算每天的平均流量 =====
    day_avg_flow = {}
    for day_key, flows in all_day_flows.items():
        day_avg_flow[day_key] = float(np.mean(flows))

    # 找出流量最小和最大的两天
    assert len(day_avg_flow) >= 2,         f"collect_temporal_basis_weights: need at least 2 days, got {len(day_avg_flow)} — check day grouping logic"
    sorted_days = sorted(day_avg_flow.items(), key=lambda x: x[1])
    min_day = sorted_days[0][0]
    min_flow = sorted_days[0][1]
    max_day = sorted_days[-1][0]
    max_flow = sorted_days[-1][1]

    if _weekday_skipped_batches > 0 or _weekend_skipped_batches > 0:
        pass  # Class imbalance warning (optional debug info)

    return {
        'per_timestep': per_timestep,
        'per_sample': all_temporal_weights,
        'time_of_day': all_time_of_day,
        'time_of_day_indices': tod_indices_all,
        'day_of_week': all_day_of_week,
        'counts_per_timestep': counts_per_timestep,
        'per_timestep_weekday': per_timestep_weekday,
        'per_timestep_weekend': per_timestep_weekend,
        'counts_weekday': counts_weekday,
        'counts_weekend': counts_weekend,
        'flow_per_slot_weekday': flow_per_slot_weekday,
        'flow_per_slot_weekend': flow_per_slot_weekend,
        'day_avg_flow': day_avg_flow,
        'min_day': min_day,
        'max_day': max_day,
        'min_flow': min_flow,
        'max_flow': max_flow,
        'all_day_tod': all_day_tod,
        'all_day_weights': all_day_weights,
        'all_day_flows': all_day_flows,
    }


def get_time_label_for_viz(i, dataset_type):
    """获取时间标签"""
    if dataset_type == '8:00-20:00':
        hour = 8 + i // 2
        minute = (i % 2) * 30
    else:
        hour = i // 2
        minute = (i % 2) * 30
    return f"{hour:02d}:{minute:02d}"


def get_hour_decimal_for_viz(i, dataset_type):
    """获取小时（小数）"""
    if dataset_type == '8:00-20:00':
        return 8.0 + i * 0.5
    else:
        return i * 0.5


def save_temporal_weights_json(temporal_weights, output_dir, dataset_info, args):
    """保存权重到 JSON"""
    actual_time_slots = dataset_info['time_slots']
    dataset_type = dataset_info['type']

    weights_data = {
        'metadata': {
            'dataset': args.data,
            'n_temporal': int(temporal_weights['per_timestep'].shape[1]),
            'time_slots': actual_time_slots,
            'dataset_type': dataset_type,
            'time_interval_minutes': 30,
        },
        'weights_per_timestep': {},
    }

    for i in range(actual_time_slots):
        if temporal_weights['counts_per_timestep'][i] > 0:
            time_label = get_time_label_for_viz(i, dataset_type)
            weights_data['weights_per_timestep'][time_label] = {
                'slot_index': int(i),
                'weights': temporal_weights['per_timestep'][i].tolist(),
                'sample_count': int(temporal_weights['counts_per_timestep'][i]),
            }

    with open(output_dir / 'temporal_weights.json', 'w') as f:
        json.dump(weights_data, f, indent=2, ensure_ascii=False)


def save_temporal_weights_csv(temporal_weights, output_dir, dataset_info, args):
    """保存权重到 CSV"""
    actual_time_slots = dataset_info['time_slots']
    dataset_type = dataset_info['type']
    n_temporal = temporal_weights['per_timestep'].shape[1]

    rows = []
    for i in range(actual_time_slots):
        if temporal_weights['counts_per_timestep'][i] > 0:
            time_label = get_time_label_for_viz(i, dataset_type)
            hour_decimal = get_hour_decimal_for_viz(i, dataset_type)

            row = {
                'slot_index': i,
                'time_label': time_label,
                'hour_decimal': hour_decimal,
                'sample_count': int(temporal_weights['counts_per_timestep'][i]),
            }
            for j in range(n_temporal):
                row[f'prototype_{j}'] = float(temporal_weights['per_timestep'][i, j])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'temporal_weights_per_timestep.csv', index=False)

    # 保存样本映射
    assert len(temporal_weights['time_of_day']) > 0, \
        "save_temporal_weights_csv: temporal_weights['time_of_day'] is empty (already asserted above but re-checked)"
    df_mapping = pd.DataFrame({
        'sample_index': range(len(temporal_weights['time_of_day'])),
        'time_of_day_value': temporal_weights['time_of_day'],
        'time_slot_index': temporal_weights['time_of_day_indices'],
        'day_of_week': temporal_weights['day_of_week'],
    })
    df_mapping.to_csv(output_dir / 'sample_time_mapping.csv', index=False)


def plot_temporal_basis_line(temporal_weights, output_dir, dataset_info, args):
    """绘制 Temporal Basis 折线图

    每个 temporal prototype 一条折线，横轴是一天中的时间槽，纵轴是该 prototype
    的平均 attention weight（softmax 输出）。

    训练良好的 temporal basis 会呈现清晰的「分工」：每个 prototype 倾向于在不同
    时间段被激活（例如早高峰 / 晚高峰 / 夜间低谷）。
    """
    actual_time_slots = dataset_info['time_slots']
    dataset_type = dataset_info['type']

    per_timestep = temporal_weights['per_timestep']  # (T, n_temporal)
    counts = temporal_weights['counts_per_timestep']  # (T,)
    n_temporal = per_timestep.shape[1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只绘制有样本的时间槽
    active_mask = counts > 0
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x_hours = np.array([get_hour_decimal_for_viz(i, dataset_type) for i in range(actual_time_slots)])
    x_labels = [get_time_label_for_viz(i, dataset_type) for i in active_indices]

    # 用 tab20 调色板，n_temporal 一般不超过 16
    cmap = plt.get_cmap('tab20')
    colors = [cmap(j % 20) for j in range(n_temporal)]

    for j in range(n_temporal):
        y = per_timestep[active_indices, j]
        ax.plot(
            x_hours[active_indices], y,
            marker='o', markersize=4,
            linewidth=1.8,
            color=colors[j],
            label=f'T{j}',
            alpha=0.85,
        )

    ax.set_xlabel('Time of day', fontsize=12)
    ax.set_ylabel('Avg. prototype weight (softmax)', fontsize=12)
    ax.set_title(
        f'Temporal Basis — Per-Prototype Average Attention ({args.data})',
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_temporal, 8), frameon=False, fontsize=9,
    )

    # X 轴：用实际时间字符串做 tick label
    step = max(1, len(active_indices) // 12)
    tick_positions = x_hours[active_indices][::step]
    tick_labels = x_labels[::step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    fig.tight_layout()
    save_path = output_dir / 'temporal_basis_line.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def collect_and_visualize_embeddings(model, test_loader, scaler, device, args, dataset_info, output_dir):
    """收集并可视化 TID/DIW 嵌入"""
    model.eval()

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_tid = model.if_time_in_day
    has_diw = model.if_day_in_week

    assert has_tid or has_diw,         "collect_and_visualize_embeddings: model must have either TID or DIW embeddings (if_time_in_day or if_day_in_week)" 

    tid_dim = model.temp_dim_tid if has_tid else 0
    diw_dim = model.temp_dim_diw if has_diw else 0

    all_tid_emb, all_diw_emb = [], []
    all_tod_values, all_dow_values = [], []

    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            _, _, time_in_day_emb, day_in_week_emb, _ = model._get_embeddings({'X': X})

            if has_tid:
                assert time_in_day_emb is not None,                     "collect_and_visualize_embeddings: time_in_day_emb is None"
                tid_emb_avg = time_in_day_emb[:, -1, :, :].mean(dim=1).cpu().numpy()
                all_tid_emb.append(tid_emb_avg)
                all_tod_values.append(X[:, -1, 0, 1].cpu().numpy())

            if has_diw:
                assert day_in_week_emb is not None,                     "collect_and_visualize_embeddings: day_in_week_emb is None"
                diw_emb_avg = day_in_week_emb[:, -1, :, :].mean(dim=1).cpu().numpy()
                all_diw_emb.append(diw_emb_avg)
                all_dow_values.append(X[:, -1, 0, 2].cpu().numpy())

    if has_tid:
        assert len(all_tid_emb) > 0,             "collect_and_visualize_embeddings: no TID embeddings collected — check test_loader data" 
        all_tid_emb = np.concatenate(all_tid_emb, axis=0)
        all_tod_values = np.concatenate(all_tod_values, axis=0)

        tid_df = pd.DataFrame(all_tid_emb, columns=[f'tid_dim_{i}' for i in range(tid_dim)])
        tid_df['time_of_day'] = all_tod_values
        hours = all_tod_values * 24
        tid_df['hour'] = hours.astype(int)
        tid_df['minute'] = ((hours - tid_df['hour']) * 60).astype(int)
        tid_df['time_label'] = tid_df.apply(lambda r: f"{int(r['hour']):02d}:{int(r['minute']):02d}", axis=1)
        tid_df.to_csv(output_dir / 'tid_embeddings.csv', index=False)



    if has_diw:
        assert len(all_diw_emb) > 0,             "collect_and_visualize_embeddings: no DIW embeddings collected — check test_loader data" 
        all_diw_emb = np.concatenate(all_diw_emb, axis=0)
        all_dow_values = np.concatenate(all_dow_values, axis=0)

        diw_df = pd.DataFrame(all_diw_emb, columns=[f'diw_dim_{i}' for i in range(diw_dim)])
        diw_df['day_of_week'] = all_dow_values
        day_indices = (all_dow_values * 7).astype(int).clip(0, 6)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        diw_df['day_name'] = [day_names[i] for i in day_indices]
        diw_df.to_csv(output_dir / 'diw_embeddings.csv', index=False)



    n_tod = len(all_tod_values)
    n_dow = len(all_dow_values)
    embedding_metadata = {
        'tid_dim': tid_dim if has_tid else 0,
        'diw_dim': diw_dim if has_diw else 0,
        'total_samples': n_tod if n_tod > 0 else n_dow,
        'dataset_type': dataset_info['type'] if dataset_info else 'unknown',
    }
    with open(output_dir / 'embedding_metadata.json', 'w') as f:
        json.dump(embedding_metadata, f, indent=2)


def run_all_visualizations(model, test_loader, scaler, device, args, save_dir):
    """统一入口：根据模型配置决定生成哪些可视化

    Args:
        model: 已训练好的模型（inference mode）
        test_loader: 测试集 DataLoader
        scaler: StandardScaler（用于反归一化）
        device: torch device
        args: argparse Namespace（包含 model, data, use_proto, use_spatio, use_temporal 等）
        save_dir: 保存目录（一般是 results/<data>/<model>_<timestamp>/）
    """
    is_pstid = args.model.upper() == 'PSTID'

    if is_pstid:
        # PSTID: TID/DIW 嵌入始终生成；proto 可视化根据 use_proto 判断
        collect_and_visualize_embeddings(
            model, test_loader, scaler, device, args, None,
            Path(save_dir) / 'visualization_temporal',
        )

        # Proto 可视化（只有完整 proto 配置才生成）
        if args.use_proto and args.use_spatio and args.use_temporal:
            run_visualization(model, test_loader, scaler, device, args, save_dir)
            run_temporal_basis_visualization(model, test_loader, scaler, device, args, save_dir)
    else:
        # 其他模型：只有完整 proto 配置才生成可视化
        full_proto = bool(args.use_proto) and bool(args.use_spatio) and bool(args.use_temporal)
        if not full_proto:
            pass  # Ablation run warning (optional debug info)
        else:
            run_visualization(model, test_loader, scaler, device, args, save_dir)


# ============================================================================
# 独立 CLI：从 checkpoint 加载并可视化
# ============================================================================

def _load_model_from_checkpoint(checkpoint_path, data_dir, data_name, gpu, logger_):
    """从 checkpoint 加载模型

    Returns:
        model, test_loader, scaler, device, args
    """
    from torch.utils.data import DataLoader
    from model_builder import build_model
    from utils.data_utils import load_dataset, PSTIDDataset, StandardScaler

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    assert 'config' in checkpoint,         f"_load_model_from_checkpoint: 'config' key missing from checkpoint {checkpoint_path}"
    config = checkpoint['config']

    # build_model 接收 argparse.Namespace
    args = argparse.Namespace(**config)

    args.data_dir = data_dir
    if not hasattr(args, 'gpu'):
        raise AttributeError("_load_model_from_checkpoint: args.gpu not found in checkpoint config")
    if not hasattr(args, 'seed'):
        raise AttributeError("_load_model_from_checkpoint: args.seed not found in checkpoint config")
    if not hasattr(args, 'batch_size'):
        raise AttributeError("_load_model_from_checkpoint: args.batch_size not found in checkpoint config")

    # 设置 logger
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.INFO)
    if not train_logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        train_logger.addHandler(handler)
    logger_.handlers = train_logger.handlers
    logger_.setLevel(logging.INFO)
    logger_.propagate = False

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
    logger_.info(f"Device: {device}")
    logger_.info(f"Loaded checkpoint: {checkpoint_path}")
    logger_.info(f"Config model: {args.model}, data: {args.data}")

    # 加载数据
    data_name_eff = data_name or args.data
    logger_.info(f"Loading data: {data_name_eff}")
    dataset = load_dataset(args.data_dir, data_name_eff, seed=args.seed)
    num_nodes = dataset['train']['X'].shape[2]
    adj_mx = torch.from_numpy(dataset['adj_mx']).float().to(device)

    # 构建模型
    logger_.info("Building model...")
    model = build_model(args.model, args, num_nodes, adj_mx, device)

    # 加载权重（去掉多余 buffer）
    state_dict = checkpoint['model_state_dict']
    for key in ['zero_ratio']:
        if key in state_dict and not any(k.startswith(f'proto_module.{key}') for k in model.state_dict()):
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # scaler
    scaler = StandardScaler(mean=checkpoint['scaler_mean'], std=checkpoint['scaler_std'])

    # test loader
    test_dataset = PSTIDDataset(dataset, 'test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    return model, test_loader, scaler, device, args, dataset


def create_cli_parser():
    """创建独立 CLI 参数解析器"""
    parser = argparse.ArgumentParser(
        description='PSTGN 可视化：从 checkpoint 加载模型并生成所有可视化',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径，如 results/cq/PSTID_xxx/checkpoint/best_model.pth')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='可视化输出目录。默认为 <checkpoint_dir>/../visualization_standalone')
    parser.add_argument('--data_dir', type=str, default='data/processed/')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集名称（默认从 config 中读取）')
    parser.add_argument('--gpu', type=int, default=0)
    return parser


def main():
    """独立可视化 CLI 入口"""
    cli_args = create_cli_parser().parse_args()

    if not os.path.exists(cli_args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {cli_args.checkpoint}")

    # 默认 save_dir：<run_dir>/visualization_standalone
    if cli_args.save_dir is None:
        ckpt_path = Path(cli_args.checkpoint).resolve()
        run_dir = ckpt_path.parent.parent  # checkpoint/ → run_dir
        cli_args.save_dir = str(run_dir / 'visualization_standalone')

    save_dir = Path(cli_args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model, test_loader, scaler, device, args, dataset = _load_model_from_checkpoint(
        cli_args.checkpoint, cli_args.data_dir, cli_args.data, cli_args.gpu, logger,
    )

    # 跑所有可视化
    run_all_visualizations(model, test_loader, scaler, device, args, save_dir)

    for f in sorted(save_dir.rglob('*')):
        if f.is_file():
            print(f"  - {f.relative_to(save_dir)}")


if __name__ == '__main__':
    main()
