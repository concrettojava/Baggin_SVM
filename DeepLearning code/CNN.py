#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 读取CSV文件
data = pd.read_csv('taiwan_bankrupt_data.csv').dropna()

# 分割特征和标签
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #卷积核感受野为5，并将原始数据通道扩展为16
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5,padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        #感受野升至3*5=16（根据原始数据维度而定），16维数据映射至32
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(1408, 128)  
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加输入通道维度
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
#         x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 初始化模型和优化器
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()

# 记录指标的列表
train_acc_list = []
train_precision_list = []
train_recall_list = []
train_f1_macro_list = []
train_f1_micro_list = []
train_specificity_list = []
train_npv_list = []

val_acc_list = []
val_precision_list = []
val_recall_list = []
val_f1_macro_list = []
val_f1_micro_list = []
val_specificity_list = []
val_npv_list = []

# 训练模型
num_epochs = 2000
#不平衡数据的两类选择置信度，大于0.25的概率即视为阳性样本（小类）
threshold = 0.25
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0, last_epoch=-1)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    predictions = (torch.sigmoid(outputs) >= threshold).float().squeeze()
    loss = criterion(outputs, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()
    scheduler.step()

    # 在训练集上计算指标
    model.eval()
    with torch.no_grad():
        train_preds = (torch.sigmoid(model(X_train)) >= threshold).float().squeeze().detach().numpy()
        train_acc = accuracy_score(y_train.detach().numpy(), train_preds)
        train_precision = precision_score(y_train.detach().numpy(), train_preds, zero_division=0)
        train_recall = recall_score(y_train.detach().numpy(), train_preds)
        train_f1_macro = f1_score(y_train.detach().numpy(), train_preds, average='macro')
        train_f1_micro = f1_score(y_train.detach().numpy(), train_preds, average='micro')
        tn, fp, fn, tp = confusion_matrix(y_train.detach().numpy(), train_preds).ravel()
        train_specificity = tn / (tn + fp)
        if tn == 0:
            train_npv = 0
        else:
            train_npv = tn / (tn + fn)

    # 在验证集上计算指标
    with torch.no_grad():
        val_preds = (torch.sigmoid(model(X_val)) >= threshold).float().squeeze().detach().numpy()
        val_acc = accuracy_score(y_val.detach().numpy(), val_preds)
        val_precision = precision_score(y_val.detach().numpy(), val_preds, zero_division=0)
        val_recall = recall_score(y_val.detach().numpy(), val_preds)
        val_f1_macro = f1_score(y_val.detach().numpy(), val_preds, average='macro')
        val_f1_micro = f1_score(y_val.detach().numpy(), val_preds, average='micro')
        tn, fp, fn, tp = confusion_matrix(y_val.detach().numpy(), val_preds).ravel()
        val_specificity = tn / (tn + fp)
        if tn == 0:
            val_npv = 0
        else:
            val_npv = tn / (tn + fn)

#     if val_acc == 1:
#         break

    # 记录指标
    train_acc_list.append(train_acc)
    train_precision_list.append(train_precision)
    train_recall_list.append(train_recall)
    train_f1_macro_list.append(train_f1_macro)
    train_f1_micro_list.append(train_f1_micro)
    train_specificity_list.append(train_specificity)
    train_npv_list.append(train_npv)

    val_acc_list.append(val_acc)
    val_precision_list.append(val_precision)
    val_recall_list.append(val_recall)
    val_f1_macro_list.append(val_f1_macro)
    val_f1_micro_list.append(val_f1_micro)
    val_specificity_list.append(val_specificity)
    val_npv_list.append(val_npv)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')
    print(f'Train - Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
    print(f'Train - F1 (Macro): {train_f1_macro:.4f}, F1 (Micro): {train_f1_micro:.4f}')
    print(f'Train - Specificity: {train_specificity:.4f}, NPV: {train_npv:.4f}')
    print(f'Val - Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
    print(f'Val - F1 (Macro): {val_f1_macro:.4f}, F1 (Micro): {val_f1_micro:.4f}')
    print(f'Val - Specificity: {val_specificity:.4f}, NPV: {val_npv:.4f}')
    print('---')

# 将训练集和测试集的指标保存为DataFrame
train_metrics = pd.DataFrame({
    'Accuracy': train_acc_list,
    'Precision': train_precision_list,
    'Recall': train_recall_list,
    'F1 Macro': train_f1_macro_list,
    'F1 Micro': train_f1_micro_list,
    'Specificity': train_specificity_list,
    'NPV': train_npv_list
})

val_metrics = pd.DataFrame({
    'Accuracy': val_acc_list,
    'Precision': val_precision_list,
    'Recall': val_recall_list,
    'F1 Macro': val_f1_macro_list,
    'F1 Micro': val_f1_micro_list,
    'Specificity': val_specificity_list,
    'NPV': val_npv_list
})

# 打印训练集的指标
print("Train Metrics:")
print(train_metrics)

# 打印验证集的指标
print("Validation Metrics:")
print(val_metrics)

# 绘制指标曲线
epochs = range(1, num_epochs+1)

plt.figure(figsize=(15, 8))

plt.subplot(2, 4, 1)
plt.plot(epochs, train_acc_list, 'b', label='Train')
plt.plot(epochs, val_acc_list, 'r', label='Val')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(epochs, train_precision_list, 'b', label='Train')
plt.plot(epochs, val_precision_list, 'r', label='Val')
plt.title('Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(epochs, train_recall_list, 'b', label='Train')
plt.plot(epochs, val_recall_list, 'r', label='Val')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(epochs, train_npv_list, 'b', label='Train')
plt.plot(epochs, val_npv_list, 'r', label='Val')
plt.title('NPV')
plt.xlabel('Epochs')
plt.ylabel('NPV')
plt.legend()

plt.subplot(2, 4, 5)
plt.plot(epochs, train_specificity_list, 'b', label='Train')
plt.plot(epochs, val_specificity_list, 'r', label='Val')
plt.title('Specificity')
plt.xlabel('Epochs')
plt.ylabel('Specificity')
plt.legend()

plt.subplot(2, 4, 6)
plt.plot(epochs, train_f1_macro_list, 'b', label='Train')
plt.plot(epochs, val_f1_macro_list, 'r', label='Val')
plt.title('F1 Macro')
plt.xlabel('Epochs')
plt.ylabel('F1 Macro')
plt.legend()

plt.subplot(2, 4, 7)
plt.plot(epochs, train_f1_micro_list, 'b', label='Train')
plt.plot(epochs, val_f1_micro_list, 'r', label='Val')
plt.title('F1 Micro')
plt.xlabel('Epochs')
plt.ylabel('F1 Micro')
plt.legend()

#显示训练曲线
plt.tight_layout()
plt.show()


# In[10]:


val_metrics.iloc[-1]

