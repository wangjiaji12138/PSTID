import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 生成5个节点的历史数据 (48个时间步)
# 使用不同的趋势模式: 正弦波、上升、下降、周期波动、随机游走
n_timesteps = 48
n_nodes = 5

# 历史数据基础趋势
history_trends = [
    lambda t: 50 + 20 * np.sin(2 * np.pi * t / 48) + 5 * np.cos(4 * np.pi * t / 48),  # N1: 正弦波动
    lambda t: 30 + 0.3 * t + 3 * np.sin(2 * np.pi * t / 24),  # N2: 缓慢上升
    lambda t: 80 - 0.4 * t + 4 * np.random.randn(len(t)) if hasattr(t, '__len__') else 80 - 0.4 * t + 4 * np.sin(2 * np.pi * t / 20),  # N3: 下降趋势
    lambda t: 40 + 15 * np.sin(2 * np.pi * t / 12) * (t/48),  # N4: 增强波动
    lambda t: 60 + 10 * np.cos(2 * np.pi * t / 16) + 8 * np.sin(2 * np.pi * t / 8),  # N5: 复杂周期
]

# 未来数据基础趋势 (接续历史但略有不同)
future_trends = [
    lambda t: 55 + 15 * np.sin(2 * np.pi * t / 48 + 0.5) + 4 * np.cos(4 * np.pi * t / 48),  # N1: 偏移正弦
    lambda t: 45 + 0.2 * t + 3 * np.sin(2 * np.pi * t / 24 + 1),  # N2: 稍缓上升
    lambda t: 65 - 0.35 * t + 5 * np.sin(2 * np.pi * t / 18 + 0.3),  # N3: 略陡下降
    lambda t: 55 - 10 * np.sin(2 * np.pi * t / 10) * ((t+48)/48),  # N4: 反向增强
    lambda t: 50 + 12 * np.cos(2 * np.pi * t / 14 + 1.5) + 6 * np.sin(2 * np.pi * t / 7),  # N5: 相位偏移
]

# 生成时间步索引
t = np.arange(n_timesteps)
t_future = np.arange(n_timesteps)

# 存储结果
data = {}

# 生成历史值
for i in range(n_nodes):
    trend_func = history_trends[i]
    base = trend_func(t)
    # 添加毛刺噪声 (较高频率噪声)
    noise = np.random.randn(n_timesteps) * 3  # 基础噪声
    spike = np.random.choice([0, 0, 0, 0, np.random.uniform(-8, 8)], n_timesteps)  # 偶尔的尖峰
    data[f'N{i+1}历史值'] = base + noise + spike

# 生成未来值
for i in range(n_nodes):
    trend_func = future_trends[i]
    base = trend_func(t_future)
    # 添加毛刺噪声
    noise = np.random.randn(n_timesteps) * 3
    spike = np.random.choice([0, 0, 0, 0, np.random.uniform(-8, 8)], n_timesteps)
    data[f'N{i+1}未来值'] = base + noise + spike

# 创建DataFrame
df = pd.DataFrame(data)

# 保存
df.to_csv('/data3/wangjiaji/PSTID/raw_data_vis.csv', index=False)

print("CSV已生成: raw_data_vis.csv")
print(f"形状: {df.shape}")
print("\n前5行:")
print(df.head())
print("\n统计信息:")
print(df.describe())

# =============================================================================
# 可视化: 历史值和未来值对比
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
node_names = ['N1', 'N2', 'N3', 'N4', 'N5']

# 左图: 历史值
ax1 = axes[0]
for i in range(n_nodes):
    ax1.plot(t, df[f'N{i+1}历史值'], label=node_names[i], color=colors[i], linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Time Step', fontsize=11)
ax1.set_ylabel('Value', fontsize=11)
ax1.set_title('Historical Values (History)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, n_timesteps - 1)

# 右图: 未来值
ax2 = axes[1]
for i in range(n_nodes):
    ax2.plot(t_future, df[f'N{i+1}未来值'], label=node_names[i], color=colors[i], linewidth=1.5, alpha=0.8)
ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Value', fontsize=11)
ax2.set_title('Future Values (Forecast)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, n_timesteps - 1)

plt.tight_layout()
plt.savefig('/data3/wangjiaji/PSTID/可视化结果/历史值和未来值对比.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n可视化已保存: 历史值和未来值对比.png")
