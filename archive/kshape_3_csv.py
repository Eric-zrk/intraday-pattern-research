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
DATA_PATH = 'data/raw/REPLACE_WITH_YOUR_FILE'  # 数据文件路径
START_DATE = "2015-01-04"   # 起始日期
END_DATE = "2023-10-31"     # 结束日期
SYMBOLS = ["IM"]            # 合约代码（这里只有IC）
N_CLUSTERS = 3              # 聚类数量：上行、下行、震荡
TRADING_MINUTES = 240       # 每个交易日的有效分钟数（上午120 + 下午120）

# ==============================
# 2. 读取并预处理数据：从原始CSV中提取每个交易日的标准化价格序列
# ==============================
print("正在加载数据...")
df = pd.read_csv(DATA_PATH)  # 读取原始数据

# 将字符串格式的时间列 'trade_time' 转换为 datetime 类型，并重命名为 'datetime'
df['datetime'] = pd.to_datetime(df['trade_time'])
df['symbol'] = 'IM'  # 手动添加合约标识（因为文件名是IC8888，代表IC主力连续）

# 按日期范围过滤数据
df = df[(df['datetime'] >= pd.to_datetime(START_DATE)) &
        (df['datetime'] <= pd.to_datetime(END_DATE))].copy()

# 只保留目标合约（虽然只有IC，但保持通用性）
df = df[df['symbol'].isin(SYMBOLS)].copy()

# 提取日期部分（不含时间），用于按天分组
df['date'] = df['datetime'].dt.date

# 存储所有有效交易日的价格序列（标准化后）
all_series = []
# 存储对应的元信息（日期、合约）
meta_info = []

# 按 (date, symbol) 分组，逐日处理
for (date, symbol), group in df.groupby(['date', 'symbol']):
    # 提取上午交易时段：09:30–11:30（共120分钟）
    morning = group[
        (group['datetime'].dt.time >= pd.to_datetime('09:30').time()) &
        (group['datetime'].dt.time <= pd.to_datetime('11:30').time())
    ]
    # 提取下午交易时段：13:00–15:00（共120分钟）
    afternoon = group[
        (group['datetime'].dt.time >= pd.to_datetime('13:00').time()) &
        (group['datetime'].dt.time <= pd.to_datetime('15:00').time())
    ]
    # 合并上午和下午数据，并按时间排序
    day_data = pd.concat([morning, afternoon]).sort_values('datetime')
    
    # 如果当天有效数据点少于240分钟的90%（即<216分钟），跳过该日
    if len(day_data) < TRADING_MINUTES * 0.9:
        continue
        
    # 构造完整的240分钟时间索引（跳过午休11:30–13:00）
    full_index = (
        pd.date_range(start=f"{date} 09:30", periods=120, freq='1T')  # 上午120分钟
        .union(pd.date_range(start=f"{date} 13:00", periods=120, freq='1T'))  # 下午120分钟
    )
    
    # 将当天的收盘价对齐到完整时间索引上，缺失值用插值+前后填充补全
    close_series = day_data.set_index('datetime')['close'].reindex(full_index)
    close_series = close_series.interpolate().fillna(method='bfill').fillna(method='ffill')
    
    # 再次检查：确保长度正好是240，且无缺失值
    if len(close_series) != TRADING_MINUTES or close_series.isna().any():
        continue
        
    # 转为浮点数组
    prICes = close_series.values.astype(np.float32)
    
    # 计算当日均值和标准差，用于标准化
    daily_mean = np.mean(prICes)
    daily_std = np.std(prICes)
    if daily_std == 0:  # 防止除零（价格完全不变的情况）
        continue
    
    # 标准化公式：(prICe / mean - 1) / std
    # 目的：消除量纲，聚焦相对波动形态（而非绝对价格水平）
    normalized = (prICes / daily_mean - 1) / daily_std
    all_series.append(normalized)
    meta_info.append((date, symbol))

print(f"共提取 {len(all_series)} 个有效交易日序列。")
if len(all_series) == 0:
    raise ValueError("没有有效数据！请检查日期范围或数据格式。")

# 将列表转为 NumPy 数组，形状为 (样本数, 时间步长, 1)
# tslearn 要求输入为三维：(n_samples, n_timesteps, n_features)
X = np.array(all_series).reshape(len(all_series), TRADING_MINUTES, 1)

# ==============================
# 3. 构造3个自定义初始簇中心（引导聚类方向）
# ==============================
# 创建一个从0到1的线性时间轴（共240个点）
t = np.linspace(0, 1, TRADING_MINUTES)

initial_centers = []

# 1. 上行趋势：价格随时间单调上升
c1 = t
c1 = (c1 - c1.mean()) / c1.std()  # 标准化（z-score）
initial_centers.append(c1)

# 2. 下行趋势：价格随时间单调下降
c2 = -t
c2 = (c2 - c2.mean()) / c2.std()
initial_centers.append(c2)

# 3. 震荡行情：高频小幅波动（用正弦波模拟无趋势的来回震荡）
c3 = np.sin(8 * np.pi * t)  # 8π 表示4个完整周期（240分钟内震荡4次）
c3 = (c3 - c3.mean()) / c3.std()
initial_centers.append(c3)

# 转换为 tslearn 所需的三维格式：(n_clusters, n_timesteps, 1)
init_centers = np.array(initial_centers).reshape(N_CLUSTERS, TRADING_MINUTES, 1)

# 可视化初始中心（便于确认设计是否合理）
plt.figure(figsize=(12, 4))
titles = ['上行趋势 ↑', '下行趋势 ↓', '震荡 ↔']
for i in range(N_CLUSTERS):
    plt.subplot(1, 3, i+1)
    plt.plot(init_centers[i].ravel(), 'b-')  # .ravel() 展平为一维
    plt.title(titles[i])
    plt.ylim(-2, 2)  # 统一纵轴范围，便于比较
plt.suptitle('自定义初始簇中心（3类）')
plt.tight_layout()
#plt.show()

# ==============================
# 4. K-Shape 聚类（使用自定义初始中心）
# ==============================
print("正在进行 K-Shape 聚类 (K=3, 自定义初始中心)...")
# 初始化KShape模型：
# - n_clusters=3：聚成3类
# - init=init_centers：使用我们设计的初始中心
# - n_init=1：只运行一次（因为初始中心已固定）
# - random_state=42：保证结果可复现
ks = KShape(n_clusters=N_CLUSTERS, init=init_centers, n_init=1, random_state=42, verbose=False)

# 对数据 X 进行聚类，并返回每个样本的类别标签（0, 1, 2）
labels = ks.fit_predict(X)

# ==============================
# 5. 可视化聚类结果：查看每个簇的“平均形态”（即质心）
# ==============================
plt.figure(figsize=(12, 4))
for i in range(N_CLUSTERS):
    plt.subplot(1, 3, i + 1)
    centroid = ks.cluster_centers_[i].ravel()  # 获取第i类的质心
    plt.plot(centroid, color='red', linewidth=2.5)  # 用红色粗线绘制
    count = np.sum(labels == i)  # 统计该类包含多少个交易日
    plt.title(f'{titles[i]}\n(样本数: {count})', fontsize=10)
    plt.xticks([])  # 隐藏x轴刻度
    plt.yticks([])  # 隐藏y轴刻度
    plt.ylim(-3, 3)  # 统一显示范围

plt.tight_layout()
plt.suptitle('K-Shape 聚类结果 (K=3)\n标准化: (close / mean - 1) / std', y=1.02)
plt.savefig('kshape_3_clusters_trend(IM).png', dpi=300, bbox_inches='tight')  # 保存高清图
#plt.show()

# ==============================
# 6. 保存结果：输出日期与对应走势类别
# ==============================
# 从 meta_info 中提取日期（忽略symbol，因为只有IC）
dates_only = [item[0] for item in meta_info]

# 创建结果DataFrame：date（日期）、label（0=上行, 1=下行, 2=震荡）
result_df = pd.DataFrame({
    'date': dates_only,
    'label': labels
})

# 保存为CSV文件
result_df.to_csv('futures_kshape_K3_trend(IM).csv', index=False)
print("聚类结果已保存至 'futures_kshape_K3_trend(IM).csv'")
