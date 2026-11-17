# Historical research script, paths sanitized; not the supported CLI.
import pandas as pd
import os
import glob

# # 1. 配置路径
# source_dir = 'data/raw/REPLACE_WITH_YOUR_FILE'
# output_dir = 'data/raw/REPLACE_WITH_YOUR_FILE'

# # 2. 精准名单
# target_files = [
#     '8888.CCFX_2025_1_1_2025_4_30_1min.csv',
#     '8888.CCFX_2025_10_01_2025_10_31.csv',
#     '8888.CCFX_20250301_20250801_1min.csv',
#     '8888.CCFX_1m_20250901_20250930.csv',
#     '8888.CCFX_1min_20250801_20250901.csv'
# ]

# print(f"🎯 正在处理包含表头的数据并合并...")

# future_name_list = ['IC','IM','IH','IF']


# for name in future_name_list:

#     data_list = []

#     for file_name in target_files:

#         current_source_dir = os.path.join(source_dir, f'{name}8888.CCFX_1m')

#         file_path = os.path.join(current_source_dir, name + file_name)
        
#         if not os.path.exists(file_path):
#             print(f"⚠️ 文件未找到: {file_name}")
#             continue
            
#         print(f"📖 正在处理: {file_name}")
#         try:
#             # 直接读取（不加 header=None），Pandas 会自动把第一行当做表头
#             df = pd.read_csv(file_path)
            
#             # 强制把第一列重命名为 'trade_time'
#             # 无论第一列叫什么（哪怕是空字符串），columns[0] 都能定位到它
#             df.rename(columns={df.columns[0]: 'trade_time'}, inplace=True)
            
#             # 将时间列转换为标准日期格式，指定 format='mixed' 解决格式不一的 UserWarning
#             df['trade_time'] = pd.to_datetime(df['trade_time'], format='mixed')
            
#             data_list.append(df)
                
#         except Exception as e:
#             print(f"❌ 处理 {file_name} 出错: {e}")

#     # 3. 执行合并逻辑
#     if data_list:
#         # 合并
#         full_df = pd.concat(data_list, axis=0, ignore_index=True)
        
#         # 排序
#         full_df = full_df.sort_values(by='trade_time', ascending=True)
        
#         # 去重
#         full_df = full_df.drop_duplicates(subset=['trade_time'], keep='first')
        
#         # 4. 生成文件名
#         start_date = full_df['trade_time'].min().strftime('%Y%m%d')
#         end_date = full_df['trade_time'].max().strftime('%Y%m%d')
#         output_filename = f"{name}8888.CCFX_{start_date}-{end_date}.csv"
#         output_path = os.path.join(output_dir, output_filename)
        
#         # 5. 保存
#         full_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
#         print("-" * 50)
#         print(f"🎉 合并成功！")
#         print(f"📄 输出文件: {output_filename}")
#         print(f"📈 最终行数: {len(full_df)} 行")
#     else:
#         print("❌ 未能合并任何数据，请检查文件名或路径。")




# # 1. 配置路径
# # base_dir 为所有品种文件夹所在的根目录
# base_dir = 'data/raw/REPLACE_WITH_YOUR_FILE'
# output_dir = 'data/raw/REPLACE_WITH_YOUR_FILE'

# # 2. 定义需要处理的品种
# future_name_list = ['IC', 'IM', 'IH', 'IF']

# print("🔥 启动 5 年历史数据终极合并程序...")

# for name in future_name_list:
#     print(f"\n" + ">>>" * 10 + f" 正在处理品种: {name} " + "<<<" * 10)
    
#     # 动态定位子文件夹，例如 data/raw/REPLACE_WITH_YOUR_FILE
#     current_source_dir = os.path.join(base_dir, f"{name}8888.CCFX_1m")
    
#     if not os.path.exists(current_source_dir):
#         print(f"⚠️ 找不到文件夹: {current_source_dir}，跳过该品种。")
#         continue

#     # 3. 搜索该文件夹下所有的 CSV 文件
#     # 包含类似 IM8888.CCFX_2020_1min.csv 和你刚生成的 IM8888.CCFX_20250102-20251031_Merged.csv
#     csv_files = glob.glob(os.path.join(current_source_dir, f"{name}8888.CCFX_*.csv"))
    
#     if not csv_files:
#         print(f"⚠️ 在 {current_source_dir} 中没找到相关 CSV 文件。")
#         continue

#     print(f"📂 发现 {len(csv_files)} 个历史数据文件，准备合并...")
    
#     full_data_list = []

#     for file_path in csv_files:
#         file_name = os.path.basename(file_path)
#         try:
#             # 读取数据
#             df = pd.read_csv(file_path)
            
#             # 统一时间列名
#             df.rename(columns={df.columns[0]: 'trade_time'}, inplace=True)
            
#             # 转换时间格式 (mixed 模式兼容 2020-2025 可能存在的格式差异)
#             df['trade_time'] = pd.to_datetime(df['trade_time'], format='mixed')
            
#             full_data_list.append(df)
#             print(f"✅ 已加载: {file_name} ({len(df)} 行)")
            
#         except Exception as e:
#             print(f"❌ 读取 {file_name} 出错: {e}")

#     # 4. 执行大合并逻辑
#     if full_data_list:
#         print(f"🔄 正在进行全量数据拼接与全局去重...")
        
#         # 纵向合并
#         big_df = pd.concat(full_data_list, axis=0, ignore_index=True)
        
#         # 全局按时间升序排列（确保 2020 在前，2025 在后）
#         big_df = big_df.sort_values(by='trade_time', ascending=True)
        
#         # 彻底去重（防止不同年份文件或刚才合并的文件有重叠日期）
#         initial_count = len(big_df)
#         big_df = big_df.drop_duplicates(subset=['trade_time'], keep='first')
#         final_count = len(big_df)
        
#         print(f"💡 合并完成：原始 {initial_count} 行 -> 去重后 {final_count} 行")

#         # 5. 生成文件名与保存 (同时保存 CSV 和 Parquet)
#         start_date = big_df['trade_time'].min().strftime('%Y%m%d')
#         end_date = big_df['trade_time'].max().strftime('%Y%m%d')
        
#         # 保存为 Parquet (推荐量化研究使用)
#         parquet_filename = f"{name}8888_FULL_{start_date}_{end_date}.parquet"
#         parquet_path = os.path.join(output_dir, parquet_filename)
#         big_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        
#         # 同时保存一份 CSV 备用
#         csv_filename = f"{name}8888_FULL_{start_date}_{end_date}.csv"
#         csv_path = os.path.join(output_dir, csv_filename)
#         big_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

#         print(f"🚀 【终极加速版】处理成功！")
#         print(f"📦 Parquet文件: {parquet_filename} (推荐读取这个)")
#         print(f"📄 CSV文件: {csv_filename}")
#     else:
#         print(f"❌ 品种 {name} 没有可合并的数据。")

# print("\n" + "✨"*15 + " 五年历史数据大合并任务已全部完成！ " + "✨"*15)


import pandas as pd
import os

# 1. 配置文件路径
file_path = 'data/raw/REPLACE_WITH_YOUR_FILE'

def view_december_data():
    print(f"🔍 正在检索 2021年12月 的分钟线数据...")
    
    try:
        # 2. 这里的列名根据你之前的输出应该是 'trade_time'
        # 我们直接使用 filters 读取
        df_dec = pd.read_parquet(
            file_path, 
            engine='pyarrow',
            filters=[
                ('trade_time', '>=', pd.Timestamp('2021-12-09 13:30:00')),
                ('trade_time', '<=', pd.Timestamp('2021-12-10 09:59:59'))
            ]
        )
        
        # 3. 显示结果
        if df_dec.empty:
            print("⚠️ 未找到 2021年12月 的数据。请确认 'trade_time' 列是否存在。")
            # 如果报错列名不对，可以先打印出列名看看
            temp = pd.read_parquet(file_path, engine='pyarrow')
            print(f"当前文件列名为: {temp.columns.tolist()}")
        else:
            print(f"✅ 成功读取 {len(df_dec)} 行数据。")
            
            # 简单处理一下，方便阅读
            df_dec = df_dec.sort_values('trade_time')
            
            print("\n--- 12月初数据预览 ---")
            print(df_dec.head(100))
            
            print("\n--- 12月低数据预览 ---")
            print(df_dec.tail(100))
            
            # 统计一下共有多少个交易日
            days = df_dec['trade_time'].dt.date.nunique()
            print(f"\n📅 12月共有 {days} 个有数据的交易日")

    except Exception as e:
        print(f"❌ 读取出错: {e}")

if __name__ == "__main__":
    view_december_data()

# import pandas as pd
# import numpy as np
# import os

# # --- 1. 配置路径 ---
# BASE_DIR = 'data/raw/REPLACE_WITH_YOUR_FILE'
# CROSS_PATH = os.path.join(BASE_DIR, 'IC_Features_Table.parquet')
# SEQ_PATH = os.path.join(BASE_DIR, 'IC_X_sequence.npy')

# # 定义训练集时间范围
# START_DATE = pd.to_datetime('2017-11-01').date()
# END_DATE = pd.to_datetime('2021-12-08').date()

# def slice_training_data():
#     print("📖 正在加载索引数据...")
#     # 只需要读取日期列来计算索引
#     df_cross = pd.read_parquet(CROSS_PATH)
#     df_cross['date'] = pd.to_datetime(df_cross['date']).dt.date
    
#     # 1. 找到符合日期的掩码 (Mask)
#     mask = (df_cross['date'] >= START_DATE) & (df_cross['date'] <= END_DATE)
#     indices = df_cross.index[mask].tolist()
    
#     if not indices:
#         print("❌ 错误：在指定日期范围内未找到任何数据！请检查日期。")
#         return

#     print(f"🎯 找到目标范围样本数: {len(indices)} 天")
#     print(f"📅 实际起始日期: {df_cross.loc[indices[0], 'date']}")
#     print(f"📅 实际结束日期: {df_cross.loc[indices[-1], 'date']}")

#     # 2. 裁切截面特征表并保存
#     df_train_cross = df_cross.loc[indices].reset_index(drop=True)
#     train_cross_save_path = os.path.join(BASE_DIR, 'IC_Train_Features.parquet')
#     df_train_cross.to_parquet(train_cross_save_path)
#     print(f"✅ 截面特征已保存: {train_cross_save_path}")

#     # 3. 裁切时序矩阵并保存
#     # 加载巨大的 npy 文件
#     print("⏳ 正在加载并裁切大型时序矩阵 (npy)...")
#     x_seq_all = np.load(SEQ_PATH)
#     x_seq_train = x_seq_all[indices] # 使用相同的索引进行同步裁切
    
#     train_seq_save_path = os.path.join(BASE_DIR, 'IC_Train_Sequence.npy')
#     np.save(train_seq_save_path, x_seq_train)
#     print(f"✅ 时序矩阵已保存: {train_seq_save_path}")
    
#     print("\n🚀 裁切完成！你现在可以使用这两个 'Train' 开头的文件进行训练了。")

# if __name__ == "__main__":
#     slice_training_data()
