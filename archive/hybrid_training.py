# Historical research script, paths sanitized; not the supported CLI.
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- 1. 配置区 ---
device = torch.device("cpu")
print(f"🔥 当前激活设备: {device}")

# 路径配置 (请确保文件在此路径下)
SEQ_PATH = 'data/raw/REPLACE_WITH_YOUR_FILE'
CROSS_PATH = 'data/raw/REPLACE_WITH_YOUR_FILE'
LABEL_PATH = 'data/raw/REPLACE_WITH_YOUR_FILE'

# 超参数
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
HIDDEN_DIM = 64



# --- 2. 数据集定义 ---
class HybridDataset(Dataset):
    def __init__(self, seq_data, cross_data, labels):
        self.seq_data = torch.from_numpy(seq_data).float()
        self.cross_data = torch.from_numpy(cross_data).float()
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seq_data[idx], self.cross_data[idx], self.labels[idx]

# --- 3. 模型定义 ---
class PatternPredictor(nn.Module):
    def __init__(self, sectional_dim=31, seq_dim=6, hidden_dim=64, num_classes=3):
        super(PatternPredictor, self).__init__()
        self.gru = nn.GRU(seq_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.sectional_net = nn.Sequential(
            nn.Linear(sectional_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_seq, x_cross):
        r_out, _ = self.gru(x_seq)
        seq_feat = r_out[:, -1, :] 
        cross_feat = self.sectional_net(x_cross)
        combined = torch.cat((seq_feat, cross_feat), dim=1)
        return self.classifier(combined)

# --- 4. 数据对齐与加载 ---
def prepare_data():
    print("⏳ 正在加载并对齐数据，请稍候...")
    X_seq_all = np.load(SEQ_PATH)
    df_cross = pd.read_parquet(CROSS_PATH)
    df_labels = pd.read_csv(LABEL_PATH)
    
    df_cross['date'] = pd.to_datetime(df_cross['date']).dt.date
    df_labels['date'] = pd.to_datetime(df_labels['date']).dt.date
    
    # 按照日期对齐标签
    temp = pd.merge(df_cross, df_labels, on='date', how='inner')
    full_dates_list = df_cross['date'].tolist()
    date_to_idx = {d: i for i, d in enumerate(full_dates_list)}
    
    final_indices, final_labels, final_cross = [], [], []
    for _, row in temp.iterrows():
        d = row['date']
        if d in date_to_idx:
            final_indices.append(date_to_idx[d])
            final_labels.append(row['label'])
            # 截取特征列 (假设 date 是第0列)
            final_cross.append(row.values[1:-1]) 
            
    X_seq_final = X_seq_all[final_indices]
    X_cross_final = np.array(final_cross, dtype=float)
    y_final = np.array(final_labels, dtype=int)
    dates_final = np.array(temp['date'].values) # 关键：转为 numpy array 以便 mask
    
    print(f"✅ 数据对齐完成! 样本总数: {len(y_final)}")
    return X_seq_final, X_cross_final, y_final, dates_final

# --- 5. 训练主循环 ---
def train_model():
    X_seq, X_cross, y, dates = prepare_data()
    
    # 设定日期边界
    train_start, train_end = pd.to_datetime('2017-11-01').date(), pd.to_datetime('2021-12-08').date()
    val_start, val_end = pd.to_datetime('2021-12-10').date(), pd.to_datetime('2022-12-30').date()

    # 切分训练集/验证集
    t_m = (dates >= train_start) & (dates <= train_end)
    v_m = (dates >= val_start) & (dates <= val_end)
    
    train_loader = DataLoader(HybridDataset(X_seq[t_m], X_cross[t_m], y[t_m]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HybridDataset(X_seq[v_m], X_cross[v_m], y[v_m]), batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 训练样本: {len(y[t_m])} | 验证样本: {len(y[v_m])}")
    
    model = PatternPredictor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    history = {'train_loss': [], 'val_acc': []}

    print("🚀 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seq, cross, label in train_loader:
            seq, cross, label = seq.to(device), cross.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq, cross), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seq, cross, label in val_loader:
                seq, cross, label = seq.to(device), cross.to(device), label.to(device)
                _, pred = torch.max(model(seq, cross).data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        
        acc = 100 * correct / total
        history['train_loss'].append(total_loss/len(train_loader))
        history['val_acc'].append(acc)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} Val Acc: {acc:.2f}%")

    # 保存并绘图
    torch.save(model.state_dict(), 'IC_Hybrid_Model_Reproduction.pth')
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history['train_loss']); plt.title('Loss')
    plt.subplot(1,2,2); plt.plot(history['val_acc']); plt.title('Accuracy')
    plt.show()

if __name__ == "__main__":
    train_model()
