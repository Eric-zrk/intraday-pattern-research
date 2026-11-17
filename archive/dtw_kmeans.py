# Historical research script, paths sanitized; not the supported CLI.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# ==============================
# 用户配置
# ==============================
DATA_PATH = 'data/raw/REPLACE_WITH_YOUR_FILE'
START_DATE = "2005-01-04"
END_DATE = "2025-10-31"
N_CLUSTERS = 10
TRADING_MINUTES = 240

# ==============================
# 1. 加载数据并处理 trade_time
# ==============================
print("正在加载数据...")
df = pd.read_parquet(DATA_PATH)

# 检查是否存在 'trade_time' 列
if 'trade_time' not in df.columns:
    raise ValueError(f"数据中缺少 'trade_time' 列！可用列: {df.columns.tolist()}")

# 转换 trade_time 为 datetime 并设为索引
df['trade_time'] = pd.to_datetime(df['trade_time'])
df = df.set_index('trade_time').sort_index()

# 筛选日期范围
df = df.loc[START_DATE:END_DATE]

# 查找价格列（支持多种命名）
price_col = None
for col in ['close', 'Close', 'CLOSE', 'last_price', 'price']:
    if col in df.columns:
        price_col = col
        break

if price_col is None:
    raise ValueError(f"未找到价格列！可用列: {df.columns.tolist()}")

print(f"✅ 使用时间列: 'trade_time'，价格列: '{price_col}'")

# ==============================
# 2. 按交易日提取1分钟序列
# ==============================
print("正在按交易日提取1分钟收盘价序列...")

# 按日期分组（仅日期部分）
df_days = df.groupby(df.index.date)

X = []
valid_dates = []

for date, group in df_days:
    group = group.sort_index()
    if len(group) >= TRADING_MINUTES:
        prices = group[price_col].iloc[:TRADING_MINUTES].values
        if len(prices) == TRADING_MINUTES and np.all(np.isfinite(prices)):
            X.append(prices)
            valid_dates.append(date)

X = np.array(X)
print(f"✅ 共提取 {X.shape[0]} 个交易日的有效1分钟序列")

if X.shape[0] == 0:
    raise ValueError("未提取到任何有效交易日，请检查数据完整性或 TRADING_MINUTES 设置")

# ==============================
# 3. 标准化（z-score per series）
# ==============================
scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X.reshape((X.shape[0], TRADING_MINUTES, 1)))

# ==============================
# 4. DTW-KMeans 聚类（K=10）
# ==============================
print(f"正在进行 DTW-KMeans 聚类（K={N_CLUSTERS}）...")
model = TimeSeriesKMeans(
    n_clusters=N_CLUSTERS,
    metric="dtw",
    max_iter=15,
    random_state=42,
    n_jobs=-1
)
labels = model.fit_predict(X_scaled)

# ==============================
# 5. 可视化结果
# ==============================
plt.rcParams.update({
    "font.family": "SimHei",
    "axes.unicode_minus": False,
    "font.size": 9,
    "figure.figsize": (14, 16),
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

fig, axes = plt.subplots(5, 2, figsize=(14, 16))
axes = axes.flatten()

for k in range(N_CLUSTERS):
    ax = axes[k]
    idxs = np.where(labels == k)[0]
    # 绘制最多 30 条样本路径
    for i in idxs[:30]:
        ax.plot(X_scaled[i].ravel(), color='gray', alpha=0.4, linewidth=0.7)
    # 绘制 DBA 中心
    ax.plot(model.cluster_centers_[k].ravel(), color='red', linewidth=2.5)
    ax.set_title(f'簇 {k+1}（{len(idxs)} 天）', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_ylabel('标准化价格')

# 隐藏多余子图
for k in range(N_CLUSTERS, len(axes)):
    axes[k].set_visible(False)

axes[-1].set_xlabel('时间（1分钟，0~239）')
plt.tight_layout()

# ==============================
# 6. 保存结果到桌面
# ==============================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
png_path = os.path.join(desktop, "DTW_KMeans_IC8888_10簇聚类结果.png")
pdf_path = os.path.join(desktop, "DTW_KMeans_IC8888_10簇聚类结果.pdf")
csv_path = os.path.join(desktop, "IC8888_聚类标签.csv")

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format="pdf", bbox_inches='tight')

# 保存聚类标签
result_df = pd.DataFrame({'date': valid_dates, 'cluster': labels})
result_df.to_csv(csv_path, index=False)

print(f"✅ 聚类完成！结果已保存至桌面：")
print(f"   - 图像: {png_path}")
print(f"   - CSV 标签: {csv_path}")
print(f"📊 各簇天数: {np.bincount(labels)}")

plt.show()
