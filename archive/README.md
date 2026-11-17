# 原始研究脚本存档

本目录包含 8 份早期实验脚本，文件名对应见 `source_mapping.json`。文本统一为 UTF-8。

唯一主动内容改动：开头增加存档说明，并将本机 C:/D: 绝对路径字符串替换为 `data/raw/REPLACE_WITH_YOUR_FILE`，避免公开原电脑路径。不会尝试把旧脚本的独立实验结果整合成已经验证的统一实验。

| 整理后的文件 | 原用途 |
| --- | --- |
| `euclidean_distance_illustration.py` | 欧氏距离示意图 |
| `merge_data.py` | 历史 CSV 合并与 Parquet 转换实验 |
| `early_kshape_experiments.py` | 早期多簇形态实验，主要代码处于注释块 |
| `kshape_3_csv.py` | 三簇 K-Shape 的 CSV 输入版本 |
| `kshape_3_parquet.py` | 三簇 K-Shape 的 Parquet 输入版本 |
| `kshape_10.py` | 十簇实验及合并为三类的历史逻辑 |
| `dtw_kmeans.py` | DTW-KMeans 对照实验 |
| `hybrid_training.py` | 原双分支网络训练与验证脚本 |

这些是历史材料，不属于默认入口。部分脚本在导入时即读取文件、画图或写出结果；请勿批量导入或直接运行。支持的入口为仓库根目录安装后的 `intraday-patterns`。旧脚本中的不足与修正记录见 `docs/IMPLEMENTATION_NOTES.md`。
