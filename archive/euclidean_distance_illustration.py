# Historical research script, paths sanitized; not the supported CLI.
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw_path
import os


'''
# ==============================
# 1. 生成原始信号（频率递增的 chirp）
# ==============================
n = 1000
t = np.linspace(0, 1000, n)

f0 = 0.3 / 100   # 起始频率（慢）
f1 = 3.0 / 100   # 结束频率（适中）

T = t[-1]
phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / T)
original_signal = np.sin(phase)

# ==============================
# 2. 生成低频变形信号（仅前 300 点，周期 ≈ 50 点）
# ==============================
distorted_signal = np.full(n, np.nan)  # 后段为 NaN，不显示

# 前 300 点：低频三角波（周期 = 50 点）
period = 50  # 每 50 个点一个完整三角波（比之前更慢）
for i in range(300):
    cycle_pos = (i % period) / period  # 0 ~ 1
    if cycle_pos < 0.5:
        tri = 2 * cycle_pos      # 上升
    else:
        tri = 2 * (1 - cycle_pos)  # 下降
    distorted_signal[i] = tri * 2 - 1  # 映射到 [-1, 1]

# ==============================
# 3. 绘图设置
# ==============================
plt.rcParams.update({
    "font.family": "SimHei",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "lines.linewidth": 2,
    "figure.figsize": (10, 3),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(dpi=300)

# 蓝色原始信号（完整 0~1000）
ax.plot(t, original_signal, 'b-', linewidth=1.8)

# 橙色变形信号（仅前 300 点，低频三角波）
ax.plot(t, distorted_signal, 'orange', linestyle='--', linewidth=1.5)

# 可选：在橙色曲线上加稀疏标记点（每 50 点一个）
marker_indices = np.arange(0, 300, 50)
ax.scatter(t[marker_indices], distorted_signal[marker_indices],
           color='orange', s=12, zorder=5)

# 坐标轴
ax.set_xlim(0, 1000)
ax.set_ylim(-1.2, 1.2)
ax.set_xticks(range(0, 1001, 100))
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.grid(True, linestyle=':', alpha=0.6)

# 标题和标签（无图例）
ax.set_title('原始信号与前段低频变形信号', fontsize=14, fontweight='bold')
ax.set_xlabel('时间 (单位: 点)')
ax.set_ylabel('幅度')

plt.tight_layout()

# ==============================
# 4. 保存到桌面
# ==============================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
png_path = os.path.join(desktop, "原始信号_低频变形.png")
pdf_path = os.path.join(desktop, "原始信号_低频变形.pdf")

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format="pdf", bbox_inches='tight')

print(f"✅ PNG 图像已保存至：{png_path}")
print(f"✅ PDF 图像已保存至：{pdf_path}")

plt.show()'''



'''
# ==============================
# 1. 生成原始 chirp 信号（频率递增）
# ==============================
n = 1000
t = np.linspace(0, 1000, n)

f0, f1 = 0.3 / 100, 3.0 / 100  # 美观频率范围
T = t[-1]
phase_orig = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / T)
original_signal = np.sin(phase_orig)

# ==============================
# 2. 生成时间扭曲的变形信号
# ==============================
def warp_time(t):
    # 非线性扭曲：前段压缩，后段拉伸
    return np.power(t / 1000, 1.4) * 1000

t_warped = warp_time(t)
distorted_signal = np.interp(t, t_warped, original_signal)
distorted_signal += np.random.normal(0, 0.015, size=distorted_signal.shape)  # 轻微噪声

# ==============================
# 3. DTW 对齐
# ==============================
path, dist = dtw_path(original_signal, distorted_signal)
path = np.array(path)

# 提取对齐序列（长度 = L）
aligned_orig = original_signal[path[:, 0]]
aligned_dist = distorted_signal[path[:, 1]]
L = len(aligned_orig)

# ==============================
# 4. 插值回 0~1000 的统一时间轴
# ==============================
new_t = np.linspace(0, 1000, 1000)          # 目标时间轴
final_original = np.interp(new_t, np.linspace(0, 1000, L), aligned_orig)
final_distorted = np.interp(new_t, np.linspace(0, 1000, L), aligned_dist)

# ==============================
# 5. 绘图（横轴 0~1000）
# ==============================
plt.rcParams.update({
    "font.family": "SimHei",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.linewidth": 1.2,
    "figure.figsize": (10, 3),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(dpi=300)

# 蓝色原始信号（对齐后插值）
ax.plot(new_t, final_original, 'b-', linewidth=2, label='原始信号')

# 橙色对齐后信号（虚线 + 稀疏点）
ax.plot(new_t, final_distorted, 'orange', linestyle='--', linewidth=1.5, label='变形信号')
# 每隔 40 个点画一个 marker
marker_step = 40
ax.scatter(
    new_t[::marker_step],
    final_distorted[::marker_step],
    color='orange',
    s=12,
    zorder=5
)

# 坐标轴设置：严格 0~1000
ax.set_xlim(0, 1000)
ax.set_ylim(-1.2, 1.2)
ax.set_xticks(range(0, 1001, 100))
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.grid(True, linestyle=':', alpha=0.6)

# 标题
title_text = f"对齐信号 (欧几里得距离: {dist:.6f})"
ax.set_title(title_text, fontsize=14, fontweight='bold')
ax.set_xlabel('时间 (单位: 点)')
ax.set_ylabel('幅度')

plt.tight_layout()

# ==============================
# 6. 保存到桌面
# ==============================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
png_path = os.path.join(desktop, "对齐信号_时间轴1000.png")
pdf_path = os.path.join(desktop, "对齐信号_时间轴1000.pdf")

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format="pdf", bbox_inches='tight')

print(f"✅ 图像已保存至桌面：{png_path}")
print(f"📊 DTW 距离: {dist:.6f}")

plt.show()'''





