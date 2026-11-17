# Historical research script, paths sanitized; not the supported CLI.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 1. 用户配置
# ==============================
DATA_PATH = 'data/raw/REPLACE_WITH_YOUR_FILE'
START_DATE = "2017-11-01"
END_DATE = "2022-12-31"
SYMBOLS = ["IC"]
N_CLUSTERS = 10  # ← 已改为 10
TRADING_MINUTES = 240

# ==============================
# 2. 数据加载与预处理（保持不变）
# ==============================
print("正在加载数据...")
df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['trade_time'])
df['symbol'] = 'IC'
df = df[(df['datetime'] >= pd.to_datetime(START_DATE)) &
        (df['datetime'] <= pd.to_datetime(END_DATE))].copy()
df = df[df['symbol'].isin(SYMBOLS)].copy()
df['date'] = df['datetime'].dt.date

all_series = []
meta_info = []

for (date, symbol), group in df.groupby(['date', 'symbol']):
    morning = group[(group['datetime'].dt.time >= pd.to_datetime('09:30').time()) &
                    (group['datetime'].dt.time <= pd.to_datetime('11:30').time())]
    afternoon = group[(group['datetime'].dt.time >= pd.to_datetime('13:00').time()) &
                      (group['datetime'].dt.time <= pd.to_datetime('15:00').time())]
    day_data = pd.concat([morning, afternoon]).sort_values('datetime')
    
    if len(day_data) < TRADING_MINUTES * 0.9:
        continue
        
    full_index = pd.date_range(start=f"{date} 09:30", periods=120, freq='1T').union(
                 pd.date_range(start=f"{date} 13:00", periods=120, freq='1T'))
    
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
    raise ValueError("没有有效数据！")

X = np.array(all_series).reshape(len(all_series), TRADING_MINUTES, 1)

# ==============================
# 3. 构造 10 个自定义初始簇中心（新增 2 种形态）
# ==============================
t = np.linspace(0, 4 * np.pi, TRADING_MINUTES)

def normalize_shape(arr):
    return (arr - arr.mean()) / arr.std()

initial_centers = []

# 1. 单边上行 ↑
initial_centers.append(normalize_shape(t))

# 2. 单边下行 ↓
initial_centers.append(normalize_shape(-t))

# 3. V型（跌后涨）↓↑
initial_centers.append(normalize_shape(np.abs(t - 2*np.pi)))

# 4. 倒V型（涨后跌）↑↓
initial_centers.append(normalize_shape(-np.abs(t - 2*np.pi)))

# 5. 涨-跌-涨 ↑↓↑
initial_centers.append(normalize_shape(np.sin(t)))

# 6. 跌-涨-跌 ↓↑↓
initial_centers.append(normalize_shape(-np.sin(t)))

# 7. 四折 ↑↓↑↓
initial_centers.append(normalize_shape(np.sin(1.5 * t)))

# 8. 四折 ↓↑↓↑
initial_centers.append(normalize_shape(-np.sin(1.5 * t)))

# 9. 早盘脉冲（快速冲高回落）
pulse = np.exp(-0.1 * t) * np.sin(3 * t)
initial_centers.append(normalize_shape(pulse))

# 10. 尾盘拉升（后期快速上涨）
ramp = np.concatenate([np.zeros(TRADING_MINUTES // 2), 
                       np.linspace(0, 1, TRADING_MINUTES // 2)])
initial_centers.append(normalize_shape(ramp))

init_centers = np.array(initial_centers).reshape(N_CLUSTERS, TRADING_MINUTES, 1)

# 可视化初始中心
plt.figure(figsize=(20, 8))
for i in range(N_CLUSTERS):
    plt.subplot(2, 5, i+1)  # ← 改为 2 行 5 列
    plt.plot(init_centers[i].ravel(), 'b-')
    plt.title(f'Init Center {i+1}')
    plt.ylim(-2.5, 2.5)
plt.suptitle('自定义初始簇中心（10种基本形态）')
plt.tight_layout()
plt.show()

# ==============================
# 4. K-Shape 聚类（K=10）
# ==============================
print("正在进行 K-Shape 聚类 (K=10, 自定义初始中心)...")
ks = KShape(n_clusters=N_CLUSTERS, init=init_centers, n_init=1, random_state=42, verbose=False)
labels = ks.fit_predict(X)

# ==============================
# 5. 可视化聚类结果
# ==============================
plt.figure(figsize=(20, 8))
for i in range(N_CLUSTERS):
    plt.subplot(2, 5, i + 1)  # ← 改为 2 行 5 列
    centroid = ks.cluster_centers_[i].ravel()
    plt.plot(centroid, color='red', linewidth=2.5)
    count = np.sum(labels == i)
    plt.title(f'Cluster {i+1}\n(样本数: {count})', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(-3, 3)

plt.tight_layout()
plt.suptitle('K-Shape 聚类结果 (K=10, 自定义初始中心)\n标准化: (close / mean - 1) / std', y=1.02, fontsize=14)
plt.savefig('kshape_10_clusters_custom_init.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# 6. 保存结果（保留原逻辑：0/1 不变，其余设为 2）
# ==============================
dates_only = [item[0] for item in meta_info]
labels = np.array(labels)
labels_mapped = np.where((labels == 0) | (labels == 1), labels, 2)

result_df = pd.DataFrame({
    'date': dates_only,
    'label': labels_mapped
})
result_df.to_csv('futures_kshape_K10_custom_init.csv', index=False)
print("聚类结果已保存至 'futures_kshape_K10_custom_init.csv'（列：date, label；0/1 保留，其余为 2）")
