# Historical research script, paths sanitized; not the supported CLI.
# k-shape聚类

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.utils import to_time_series_dataset
import warnings
warnings.filterwarnings("ignore")


'''
# ==============================
# 1. 用户配置
# ==============================
DATA_PATH = 'data/raw/REPLACE_WITH_YOUR_FILE'  # ✅ 已修正路径
START_DATE = "2017-11-01"
END_DATE = "2022-12-31"
SYMBOLS = ["IC"]  # ✅ 只保留 IC，因为你只有 IC8888 数据
N_CLUSTERS = 8
TRADING_MINUTES = 240

# ==============================
# 2. 读取并预处理数据
# ==============================
print("正在加载数据...")
df = pd.read_csv(DATA_PATH)

# 将原始时间列 'trade_time' 转为 datetime，并命名为 'datetime'
df['datetime'] = pd.to_datetime(df['trade_time'])  # ✅ 修复：使用 to_datetime
df['symbol'] = 'IC'  # ✅ 手动添加 symbol 列

# 按日期过滤
df = df[(df['datetime'] >= pd.to_datetime(START_DATE)) &
        (df['datetime'] <= pd.to_datetime(END_DATE))].copy()

# 只保留目标合约（虽然只有 IC，但保持逻辑一致）
df = df[df['symbol'].isin(SYMBOLS)].copy()
df['date'] = df['datetime'].dt.date

all_series = []
meta_info = []

for (date, symbol), group in df.groupby(['date', 'symbol']):
    # 提取上午 09:30–11:30（120 分钟）
    morning = group[(group['datetime'].dt.time >= pd.to_datetime('09:30').time()) &
                    (group['datetime'].dt.time <= pd.to_datetime('11:30').time())]
    # 提取下午 13:00–15:00（120 分钟）
    afternoon = group[(group['datetime'].dt.time >= pd.to_datetime('13:00').time()) &
                      (group['datetime'].dt.time <= pd.to_datetime('15:00').time())]
    day_data = pd.concat([morning, afternoon]).sort_values('datetime')
    
    if len(day_data) < TRADING_MINUTES * 0.9:
        continue
        
    # 构造完整的 240 分钟时间索引（含午休）
    full_index = pd.date_range(start=f"{date} 09:30", periods=120, freq='1T').union(
                 pd.date_range(start=f"{date} 13:00", periods=120, freq='1T'))
    
    # 对齐并插值
    close_series = day_data.set_index('datetime')['close'].reindex(full_index)
    close_series = close_series.interpolate().fillna(method='bfill').fillna(method='ffill')
    
    if len(close_series) != TRADING_MINUTES or close_series.isna().any():
        continue
        
    prices = close_series.values.astype(np.float32)
    daily_mean = np.mean(prices)
    daily_std = np.std(prices)
    if daily_std == 0:
        continue
    # 标准化：(price / mean - 1) / std
    normalized = (prices / daily_mean - 1) / daily_std
    all_series.append(normalized)
    meta_info.append((date, symbol))

print(f"共提取 {len(all_series)} 个有效交易日序列。")
if len(all_series) == 0:
    raise ValueError("没有有效数据！请检查日期范围或数据格式。")

X = np.array(all_series).reshape(len(all_series), TRADING_MINUTES, 1)

# ==============================
# 3. 构造 8 个自定义初始簇中心
# ==============================
t = np.linspace(0, 4 * np.pi, TRADING_MINUTES)

initial_centers = []

# 1. 单边上行 ↑
c1 = t
c1 = (c1 - c1.mean()) / c1.std()
initial_centers.append(c1)

# 2. 单边下行 ↓
c2 = -t
c2 = (c2 - c2.mean()) / c2.std()
initial_centers.append(c2)

# 3. V型（跌后涨）↓↑
c3 = np.abs(t - 2*np.pi)
c3 = (c3 - c3.mean()) / c3.std()
initial_centers.append(c3)

# 4. 倒V型（涨后跌）↑↓
c4 = -np.abs(t - 2*np.pi)
c4 = (c4 - c4.mean()) / c4.std()
initial_centers.append(c4)

# 5. 涨-跌-涨 ↑↓↑
c5 = np.sin(t)
c5 = (c5 - c5.mean()) / c5.std()
initial_centers.append(c5)

# 6. 跌-涨-跌 ↓↑↓
c6 = -np.sin(t)
c6 = (c6 - c6.mean()) / c6.std()
initial_centers.append(c6)

# 7. 四折 ↑↓↑↓
c7 = np.sin(1.5 * t)
c7 = (c7 - c7.mean()) / c7.std()
initial_centers.append(c7)

# 8. 四折 ↓↑↓↑
c8 = -np.sin(1.5 * t)
c8 = (c8 - c8.mean()) / c8.std()
initial_centers.append(c8)

init_centers = np.array(initial_centers).reshape(N_CLUSTERS, TRADING_MINUTES, 1)

# 可视化初始中心（可选）
plt.figure(figsize=(16, 8))
for i in range(N_CLUSTERS):
    plt.subplot(2, 4, i+1)
    plt.plot(init_centers[i].ravel(), 'b-')
    plt.title(f'Init Center {i+1}')
    plt.ylim(-2, 2)
plt.suptitle('自定义初始簇中心（8种基本形态）')
plt.tight_layout()
plt.show()

# ==============================
# 4. K-Shape 聚类（K=8，自定义初始中心）
# ==============================
print("正在进行 K-Shape 聚类 (K=8, 自定义初始中心)...")
ks = KShape(n_clusters=N_CLUSTERS, init=init_centers, n_init=1, random_state=42, verbose=False)
labels = ks.fit_predict(X)

# ==============================
# 5. 可视化聚类结果
# ==============================
plt.figure(figsize=(16, 10))
for i in range(N_CLUSTERS):
    plt.subplot(2, 4, i + 1)
    centroid = ks.cluster_centers_[i].ravel()
    plt.plot(centroid, color='red', linewidth=2.5)
    count = np.sum(labels == i)
    plt.title(f'Cluster {i+1}\n(样本数: {count})', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(-3, 3)

plt.tight_layout()
plt.suptitle('K-Shape 聚类结果 (K=8, 自定义初始中心)\n标准化: (close / mean - 1) / std', y=1.02, fontsize=14)
plt.savefig('kshape_8_clusters_custom_init.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# 6. 保存结果（仅 date + label，0/1 保留，其余转为 2）
# ==============================
dates_only = [item[0] for item in meta_info]  # 只取 date

# 将 labels 中除 0 和 1 外的所有值设为 2
import numpy as np
labels = np.array(labels)
labels_mapped = np.where((labels == 0) | (labels == 1), labels, 2)

result_df = pd.DataFrame({
    'date': dates_only,
    'label': labels_mapped
})
result_df.to_csv('futures_kshape_K8_custom_init.csv', index=False)
print("聚类结果已保存至 'futures_kshape_K8_custom_init.csv'（列：date, label；0/1 保留，其余为 2）")'''


'''
# ==============================
# 6. 保存结果
# ==============================
result_df = pd.DataFrame(meta_info, columns=['date', 'symbol'])
result_df['cluster'] = labels
result_df.to_csv('futures_kshape_K8_custom_init.csv', index=False)
print("聚类结果已保存至 'futures_kshape_K8_custom_init.csv'")
'''






