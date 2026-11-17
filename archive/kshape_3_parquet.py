# Historical research script, paths sanitized; not the supported CLI.
# 导入必要的库
import pandas as pd          # 数据处理
import numpy as np           # 数值计算
import matplotlib.pyplot as plt  # 绘图
from tslearn.clustering import KShape  # 时间序列K-Shape聚类算法
import warnings
warnings.filterwarnings("ignore")  # 忽略警告（如h5py未安装等）

# ==============================
# 1. 用户配置：定义路径、时间范围、参数等
# ==============================
# 🔁 改为 .parquet 后缀
DATA_PATH = 'data/raw/REPLACE_WITH_YOUR_FILE'  # ← 注意这里是 .parquet
START_DATE = "2005-01-04"
END_DATE = "2025-10-31"
SYMBOLS = ["IC"]
N_CLUSTERS = 3
TRADING_MINUTES = 240

# ==============================
# 2. 读取并预处理数据（使用 Parquet）
# ==============================
print("正在加载数据...")
# 🔁 使用 read_parquet 替代 read_csv
df = pd.read_parquet(DATA_PATH)  # ← 关键修改！

# 确保 'trade_time' 是字符串或 datetime；如果是字符串，需转换
if df['trade_time'].dtype == 'object':
    df['datetime'] = pd.to_datetime(df['trade_time'])
else:
    df['datetime'] = df['trade_time']  # 如果已是 datetime 类型

df['symbol'] = 'IC'

# 按日期范围过滤
df = df[(df['datetime'] >= pd.to_datetime(START_DATE)) &
        (df['datetime'] <= pd.to_datetime(END_DATE))].copy()
df = df[df['symbol'].isin(SYMBOLS)].copy()
df['date'] = df['datetime'].dt.date

all_series = []
meta_info = []

for (date, symbol), group in df.groupby(['date', 'symbol']):
    morning = group[
        (group['datetime'].dt.time >= pd.to_datetime('09:30').time()) &
        (group['datetime'].dt.time <= pd.to_datetime('11:30').time())
    ]
    afternoon = group[
        (group['datetime'].dt.time >= pd.to_datetime('13:00').time()) &
        (group['datetime'].dt.time <= pd.to_datetime('15:00').time())
    ]
    day_data = pd.concat([morning, afternoon]).sort_values('datetime')
    
    if len(day_data) < TRADING_MINUTES * 0.9:
        continue
        
    full_index = (
        pd.date_range(start=f"{date} 09:30", periods=120, freq='1T')
        .union(pd.date_range(start=f"{date} 13:00", periods=120, freq='1T'))
    )
    
    close_series = day_data.set_index('datetime')['close'].reindex(full_index)
    close_series = close_series.interpolate().fillna(method='bfill').fillna(method='ffill')
    
    if len(close_series) != TRADING_MINUTES or close_series.isna().any():
        continue
        
    prices = close_series.values.astype(np.float32)
    daily_mean = np.mean(prices)
    daily_std = np.std(prices)
    if daily_std == 0:
        continue
    normalized = (prices / daily_mean - 1) / daily_std
    all_series.append(normalized)
    meta_info.append((date, symbol))

print(f"共提取 {len(all_series)} 个有效交易日序列。")
if len(all_series) == 0:
    raise ValueError("没有有效数据！请检查日期范围或数据格式。")

X = np.array(all_series).reshape(len(all_series), TRADING_MINUTES, 1)

# ==============================
# 3. 构造3个自定义初始簇中心
# ==============================
t = np.linspace(0, 1, TRADING_MINUTES)
initial_centers = []

c1 = t; c1 = (c1 - c1.mean()) / c1.std(); initial_centers.append(c1)
c2 = -t; c2 = (c2 - c2.mean()) / c2.std(); initial_centers.append(c2)
c3 = np.sin(8 * np.pi * t); c3 = (c3 - c3.mean()) / c3.std(); initial_centers.append(c3)

init_centers = np.array(initial_centers).reshape(N_CLUSTERS, TRADING_MINUTES, 1)

plt.figure(figsize=(12, 4))
titles = ['上行趋势 ↑', '下行趋势 ↓', '震荡 ↔']
for i in range(N_CLUSTERS):
    plt.subplot(1, 3, i+1)
    plt.plot(init_centers[i].ravel(), 'b-')
    plt.title(titles[i])
    plt.ylim(-2, 2)
plt.suptitle('自定义初始簇中心（3类）')
plt.tight_layout()
plt.savefig('kshape_init_centers(IC).png', dpi=300, bbox_inches='tight')
# plt.show()  # 已注释，避免卡住

# ==============================
# 4. K-Shape 聚类
# ==============================
print("正在进行 K-Shape 聚类 (K=3, 自定义初始中心)...")
ks = KShape(n_clusters=N_CLUSTERS, init=init_centers, n_init=1, random_state=42, verbose=False)
labels = ks.fit_predict(X)

# ==============================
# 5. 可视化结果
# ==============================
plt.figure(figsize=(12, 4))
for i in range(N_CLUSTERS):
    plt.subplot(1, 3, i + 1)
    centroid = ks.cluster_centers_[i].ravel()
    plt.plot(centroid, color='red', linewidth=2.5)
    count = np.sum(labels == i)
    plt.title(f'{titles[i]}\n(样本数: {count})', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(-3, 3)

plt.tight_layout()
plt.suptitle('K-Shape 聚类结果 (K=3)\n标准化: (close / mean - 1) / std', y=1.02)
plt.savefig('kshape_3_clusters_trend(IC).png', dpi=300, bbox_inches='tight')
# plt.show()

# ==============================
# 6. 保存结果为 Parquet（而非 CSV）
# ==============================
dates_only = [item[0] for item in meta_info]
result_df = pd.DataFrame({
    'date': dates_only,
    'label': labels
})

# 🔁 保存为 .parquet
output_path = 'futures_kshape_K3_trend(IC).parquet'
result_df.to_parquet(output_path, index=False)  # ← 关键修改！
print(f"聚类结果已保存至 '{output_path}'")
