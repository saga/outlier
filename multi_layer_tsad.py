import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from pyod.models.iforest import IForest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 中文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STHeiti', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子以保证结果可复现性
np.random.seed(42)
torch.manual_seed(42)

# --- 配置参数 ---
FILE_NAME = "multi_layer_tsad.py"
# 时间窗口长度 (L)，用来捕捉时序依赖关系，可选范围5-50。5-10：适合短期模式，计算快，但可能丢失长期依赖；10-30:平衡短期和长期特征，推荐一般场景；30-50：捕捉长期趋势，但是计算成本高，需要更多训练数据
WINDOW_SIZE = 10  

# 金融运营指标维度 (D)，特征越多，模型复杂度越高，训练更多训练数据防止过拟合
N_FEATURES = 10    

 # 预计的异常数据比例（用于阈值设定），异常污染率参数。预期数据中异常样本的比例，用于IForest和动态阈值计算
CONTAMINATION = 0.05

# 融合权重: Alpha + Beta = 1
# Stage 1 (IForest) 的权重，控制传统机器学习方法的影响力。Alpha越大，模型越偏向检测全局点异常（突发波动）
ALPHA = 0.4  
# Stage 2 (CNN-LSTM AE) 的权重
BETA = 0.6   
# 融合公式： S_final = ALPHA * S1_norm + BETA * S2_norm

# 模型训练参数

# 每批次训练的样本数量，可选范围16～256。16-32:训练慢，但是泛化能力强，适合数据量少的场景；32-64:平衡，推荐默认值；128-256:训练快，需要更多内存，适合大数据集
BATCH_SIZE = 64

# CNN-LSTM自编码器的训练轮数，少轮次可能欠拟合，多轮次（30-100）可能过拟合，需要配合早停机制。
EPOCHS = 10

# Adam优化器的学习率，控制参数更新步长。1e-4 - 5e-4: 训练稳定但是慢，适合精细调参；1e-3 - 5e-3: 推荐默认；5e-3 - 1e-2：训练快但可能不稳定或者发散。学习率越大，收敛越快但是可能跳过最优解
LEARNING_RATE = 1e-3
# ------------------

# ==============================================================================
# 阶段 2: CNN-LSTM 自编码器模型定义 (PyTorch)
# ==============================================================================

class CNN_LSTM_AE(nn.Module):
    """
    CNN-LSTM 混合自编码器架构用于时间序列重构
    编码器: Conv1D(局部特征提取) + LSTM(时序依赖建模)
    解码器: LSTM(时序重构) + 全连接层(特征映射)
    """
    def __init__(self, window_size, n_features, conv_units=32, lstm_units=64):
        super(CNN_LSTM_AE, self).__init__()
        self.window_size = window_size
        
        # --- 编码器 (Encoder) ---
        # Conv1d 提取局部特征: 输入 (Batch, Channels=Features, Length=Window_size)
        # conv_units 卷积核数量，可选范围16-128，越大提取更丰富的局部特征，但是计算成本高
        # kernel_size，卷积核大小，可选范围2-7，5-7适合捕捉更长的局部模式
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=conv_units, 
                               kernel_size=3, padding=1)
        
        # LSTM 层：在时间维度上捕捉依赖
        # LSTM 期望输入 (Batch, Length, Features)
        # lstm_units 隐藏层大小，可选范围32-256，越大捕捉更复杂的时序依赖，但是容易过拟合且计算慢
        self.lstm1 = nn.LSTM(input_size=conv_units, hidden_size=lstm_units, 
                              batch_first=True)
        
        # --- 解码器 (Decoder) ---
        # 用于将 LSTM 最终状态解码回序列
        # 输入大小是 lstm_units，因为它接收编码器的最终隐藏状态
        # num_layers LSTM堆叠层数
        self.decoder_lstm = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units, 
                                    num_layers=1, batch_first=True)
        
        self.output_layer = nn.Linear(lstm_units, n_features)

    def forward(self, x):
        # 1. Conv1D 输入维度转换: (B, L, D) -> (B, D, L)
        x = x.permute(0, 2, 1)
        
        # 2. 编码器 (CNN)
        x = torch.relu(self.conv1(x))
        # 维度转换: (B, Conv_Units, L) -> (B, L, Conv_Units) for LSTM
        x = x.permute(0, 2, 1) 
        
        # 3. 编码器 (LSTM)
        # out 包含所有时间步的输出，h_n 包含最后的隐藏状态
        _, (h_n, _) = self.lstm1(x)
        
        # 4. 解码器输入准备：使用最后的隐藏状态重复 WINDOW_SIZE 次
        # h_n shape: (num_layers * num_directions, Batch, Hidden_size)
        # 提取最后一个隐藏层: h_n[-1] shape (Batch, Hidden_size)
        # 展开为序列: (Batch, Length, Hidden_size)
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, self.window_size, 1)
        
        # 5. 解码器 (LSTM)
        decoded_output, _ = self.decoder_lstm(decoder_input)
        
        # 6. 输出层
        reconstruction = self.output_layer(decoded_output)
        
        # 输出形状 (Batch, Length, Features)
        return reconstruction 

# ==============================================================================
# 数据函数
# ==============================================================================

def generate_synthetic_data(length, n_features):
    """
    生成具有趋势、季节性和噪声的多元时间序列，并插入点异常和集合异常。
    
    异常类型包括: 
    1. 点异常 (Point Outliers): 局部剧烈波动 (Stage 1/IForest 敏感)
    2. 集合异常 (Collective Outliers): 持续性趋势偏移 (Stage 2/AE 敏感)
    """
    t = np.arange(length)
    
    # 基础数据：趋势 + 季节性 + 噪声
    trend = 0.005 * t.reshape(-1, 1) + np.sin(t / 20).reshape(-1, 1)
    noise = np.random.randn(length, n_features) * 0.4
    
    # 动态生成与n_features匹配的权重数组
    trend_weights = np.linspace(0.8, 2.0, n_features)
    data = trend * trend_weights + noise
    
    # 动态生成基线偏移（50-100之间均匀分布）
    baseline_offsets = np.linspace(50, 100, n_features)
    data += baseline_offsets.reshape(1, -1)
    
    # --- 插入异常 ---
    labels = np.zeros(length)
    
    # 1. 插入点异常 (在第一个可用特征上剧烈波动)
    point_start, point_end = 250, 253
    feature_idx = min(2, n_features -1) #使用Feature2或者最后一个
    data[point_start:point_end, feature_idx] += 15.0 
    labels[point_start:point_end] = 1
    
    # 2. 插入集合异常 (Feature 0 持续上升)
    collective_start, collective_end = 700, 750
    data[collective_start:collective_end, 0] += np.linspace(2, 6, 50) 
    labels[collective_start:collective_end] = 1
    
    # 3. 插入另一个情境异常 (在第二个特征上局部反向趋势)
    context_start, context_end = 400, 420
    feature_idx2 = min(1, n_features -1)
    data[context_start:context_end, feature_idx2] -= np.linspace(3, 1, 20)
    labels[context_start:context_end] = 1
    
    print(f"原始序列长度: {length}, 特征维度：{n_features} 异常点总数: {np.sum(labels)}")
    
    return data, labels

def create_windows(series, window_size):
    """将时间序列数据转换为滑动窗口数据 (N, L, D)"""
    X_window = []
    for i in range(len(series) - window_size + 1):
        X_window.append(series[i:i + window_size])
    return np.array(X_window)

def preprocess_data(series_data, window_size):
    """标准化、窗口化和展平数据"""
    # 1. 归一化
    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(series_data)
    
    # 2. 窗口化 (用于 S2)
    X_windowed = create_windows(scaled_series, window_size)
    
    # 3. 展平窗口 (用于 S1, 展平为 (N, L*D))
    n_samples, win_len, n_dims = X_windowed.shape
    X_flat = X_windowed.reshape(n_samples, win_len * n_dims)
    
    return scaled_series, X_windowed, X_flat, scaler

# ==============================================================================
# 阶段 3: 融合与决策
# ==============================================================================

def min_max_scaler(scores):
    """对异常分数进行Min-Max归一化处理到[0, 1]区间"""
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val < 1e-10:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)

def train_and_score_ae(X_windowed, win_len, n_dims):
    """训练 PyTorch CNN-LSTM AE 并计算重构误差 S2"""
    X_tensor = torch.from_numpy(X_windowed).float()
    dataloader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True)
    
    model = CNN_LSTM_AE(win_len, n_dims)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    model.train()
    for epoch in range(EPOCHS):
        for [batch] in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
    
    # 计算异常分数 S2
    model.eval()
    S2_scores = []
    with torch.no_grad():
        test_dataloader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=False)
        for [batch] in test_dataloader:
            output = model(batch)
            # 计算每个样本的平均重构误差 (S3 公式)
            mse = torch.mean((output - batch)**2, dim=(1, 2))
            S2_scores.extend(mse.cpu().numpy())

    return np.array(S2_scores)

def multi_layer_detection_pipeline(series_data):
    """执行多层异常检测策略"""
    
    # 1. 数据预处理
    scaled_series, X_windowed, X_flat, scaler = preprocess_data(series_data, WINDOW_SIZE)
    n_samples, win_len, n_dims = X_windowed.shape
    
    # --- 阶段 1: Isolation Forest (局部隔离分数 S1) ---
    print("--- 阶段 1: 训练 Isolation Forest ---")
    start_time_if = time.time()
    clf_iforest = IForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
    clf_iforest.fit(X_flat) 
    S1 = clf_iforest.decision_scores_
    print(f"IForest 训练/评分时间: {time.time() - start_time_if:.4f} 秒")
    S1_norm = min_max_scaler(S1)
    
    # --- 阶段 2: CNN-LSTM AutoEncoder (重构误差分数 S2) ---
    print("\n--- 阶段 2: 训练 CNN-LSTM 自编码器 (PyTorch) ---")
    start_time_ae = time.time()
    S2 = train_and_score_ae(X_windowed, win_len, n_dims)
    print(f"AE 训练/评分时间: {time.time() - start_time_ae:.4f} 秒")
    S2_norm = min_max_scaler(S2)
    
    # --- 阶段 3: 分数融合与决策 ---
    
    # 融合分数 S_final = ALPHA * S1_norm + BETA * S2_norm
    S_final = (ALPHA * S1_norm) + (BETA * S2_norm)
    
    # 动态阈值确定 (基于污染率)
    threshold = np.percentile(S_final, (1 - CONTAMINATION) * 100)
    
    # 最终判定
    D_final = (S_final >= threshold).astype(int)
    
    return scaled_series, S1_norm, S2_norm, S_final, D_final, threshold

# ==============================================================================
# 可视化函数
# ==============================================================================

def plot_results(series_data, labels, S1_norm, S2_norm, S_final, D_final, threshold, window_size):
    """用图表显示原始数据、各阶段分数和最终检测结果"""
    
    # 调整数据长度以匹配窗口化后的分数长度
    series_aligned = series_data[window_size - 1:]
    y_true_aligned = labels[window_size - 1:]
    
    time_index = np.arange(len(S_final))
    
    fig, axs = plt.subplots(4, 1, figsize=(18, 15), sharex=True)
    
    # --- Plot 1: 原始数据 (仅显示 Feature 0 作为代表) ---
    axs[0].plot(time_index, series_aligned[:, 0], label='Feature 0 (Original)', color='blue', alpha=0.7)
    
    # 标记真实异常点
    true_anomalies = time_index[y_true_aligned == 1]
    if len(true_anomalies) > 0:
        axs[0].scatter(true_anomalies, series_aligned[true_anomalies, 0], color='red', marker='o', s=10, label='Ground Truth Anomalies')
        
    axs[0].set_title(f'1. Feature 0 Original Series and Ground Truth Anomalies (L={len(series_aligned)})')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot 2: 归一化的 S1 和 S2 分数 ---
    axs[1].plot(time_index, S1_norm, label=f'S1 (IForest, $\\alpha$={ALPHA})', color='green', alpha=0.7)
    axs[1].plot(time_index, S2_norm, label=f'S2 (CNN-LSTM AE, $\\beta$={BETA})', color='orange', alpha=0.7)
    axs[1].set_title('2. Stage-wise Normalized Anomaly Scores ($S_1$ vs $S_2$)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot 3: 最终融合分数 S_final ---
    axs[2].plot(time_index, S_final, label='S_final (Fused Anomaly Score)', color='purple')
    axs[2].hlines(threshold, time_index[0], time_index[-1], color='red', linestyle='--', label=f'Threshold $\\phi$={threshold:.4f}')
    
    # 标记检测到的异常点 (D_final=1)
    detected_anomalies = time_index[D_final == 1]
    if len(detected_anomalies) > 0:
        axs[2].scatter(detected_anomalies, S_final[detected_anomalies], color='red', marker='x', s=50, label='Detected Anomalies')
        
    axs[2].set_title(f'3. Final Fused Anomaly Score $S_{{final}}$ and Dynamic Threshold')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 4: 结果对比 (真实标签 vs 检测标签) ---
    axs[3].plot(time_index, y_true_aligned, label='Ground Truth (1=Anomaly)', color='red', alpha=0.6)
    axs[3].plot(time_index, D_final * 0.9, label='Detected (0.9=Anomaly)', color='blue', linestyle='--')
    axs[3].set_yticks([0, 0.9])
    axs[3].set_yticklabels(['Normal (0)', 'Anomaly (1)'])
    axs[3].set_title('4. Ground Truth vs Detection Results Comparision')
    axs[3].set_xlabel('Time Step')
    axs[3].legend(loc='upper left')
    axs[3].grid(True, linestyle='--', alpha=0.6)

    output_file = 'anomaly_detection_result.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存")
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 主执行块
# ==============================================================================

if __name__ == "__main__":
    
    # --- 1. 构造相关测试数据 ---
    DATA_LENGTH = 5000
    # 数据越多，训练越充分，建议DATA_LENGTH >= WINDOW_SIZE * 20 保证足够的窗口样本

    series_data, true_labels = generate_synthetic_data(DATA_LENGTH, N_FEATURES)
    
    # --- 2. 执行多层异常检测策略 ---
    scaled_series, S1_norm, S2_norm, S_final, D_final, threshold = \
        multi_layer_detection_pipeline(series_data)
        
    # --- 3. 结果评估 (与模拟标签对比) ---
    # 窗口化对齐真实标签
    L = WINDOW_SIZE
    y_true_aligned = true_labels[L-1:]
    
    # 确保标签中存在异常 (如果不存在，AUC/F1计算会失败)
    if np.sum(y_true_aligned) > 0:
        # AUC-ROC需要原始分数S_final，F1/Precision/Recall需要最终标签D_final
        final_auc = roc_auc_score(y_true_aligned, S_final)
        final_f1 = f1_score(y_true_aligned, D_final)
        final_precision = precision_score(y_true_aligned, D_final)
        final_recall = recall_score(y_true_aligned, D_final)

        print("\n" + "="*50)
        print("最终融合模型性能 (基于对齐后的标签):")
        print(f"真实异常点数量: {np.sum(y_true_aligned)}")
        print(f"AUC-ROC: {final_auc:.4f}")
        print(f"F1-Score: {final_f1:.4f}")
        print(f"Precision: {final_precision:.4f}")
        print(f"Recall: {final_recall:.4f}")
        print("="*50)

        results_dict = {
            '真实异常点数量': np.sum(y_true_aligned),
            'AUC-ROC': final_auc,
            'F1-Score': final_f1,
            'Precision': final_precision,
            'Recall': final_recall
        }
        import json
        with open('anomaly_detection_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

    else:
        print("\n未在对齐后的序列中检测到真实异常点, 跳过性能评估. ")

    # --- 4. 用图表显示结果 ---
    plot_results(series_data, true_labels, S1_norm, S2_norm, S_final, D_final, threshold, WINDOW_SIZE)
