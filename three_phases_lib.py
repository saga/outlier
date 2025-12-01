import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from pyod.utils.utility import min_max_scaler

# --- 配置参数 ---
WINDOW_SIZE = 10  # 用于集合/情境异常检测的时间窗口长度
N_FEATURES = 5    # 假设有5个运营指标
ALPHA, BETA, GAMMA = 0.3, 0.4, 0.3 # 融合权重

def create_windows(data, window_size):
    """将时间序列数据转换为滑动窗口数据 (N, L, D)"""
    X_window = []
    for i in range(len(data) - window_size + 1):
        X_window.append(data[i:i + window_size])
    return np.array(X_window)

def preprocess(series, window_size):
    """执行鲁棒归一化和窗口化"""
    scaler = RobustScaler()
    scaled_series = scaler.fit_transform(series)
    
    X_train_windowed = create_windows(scaled_series, window_size)
    
    # 调整深度学习输入形状: (N, L, D)
    n_samples, win_len, n_dims = X_train_windowed.shape
    
    # 针对传统模型，有时需要将窗口展平为 (N, L*D)
    X_train_flat = X_train_windowed.reshape(n_samples, win_len * n_dims)
    
    return scaled_series, X_train_windowed, X_train_flat, scaler


# 阶段 3：深度学习模型 (LSTM-AutoEncoder)

def build_lstm_autoencoder(window_size, n_features):
    """构建用于时间序列重构的 LSTM 自编码器"""
    
    input_layer = Input(shape=(window_size, n_features))
    
    # 编码器 (Encoder)
    encoded = LSTM(64, activation='relu', return_sequences=False)(input_layer)
    
    # 重复向量以匹配解码器输入序列长度
    decoded = RepeatVector(window_size)(encoded)
    
    # 解码器 (Decoder)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    # 输出层，确保输出维度匹配特征数
    output_layer = TimeDistributed(Dense(n_features))(decoded)
    
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

def get_reconstruction_scores(model, X_windowed):
    """计算重构误差 S3"""
    X_pred = model.predict(X_windowed, verbose=0)
    # 计算每个窗口的 MSE
    mse = np.mean(np.square(X_windowed - X_pred), axis=(1, 2))
    return mse.flatten()


#  多层集成与最终输出。核心集成逻辑：训练三个检测器，获取各自的异常分数，归一化后加权求和。

def multi_layer_detection(series_data):
    # 1. 数据准备
    scaled_series, X_windowed, X_flat, scaler = preprocess(series_data, WINDOW_SIZE)
    
    # 确保用于点异常检测的数据长度匹配窗口化数据（为了分数对齐）
    # 由于窗口化损失了前 L-1 个点，因此我们只对可以形成完整窗口的数据点进行检测
    X_point = scaled_series[WINDOW_SIZE-1:] 

    # --- 2. 阶段 1: ECOD (点异常) ---
    # ECOD 适用于处理原始特征数据
    clf_ecod = ECOD(contamination=0.01) # contamination 用于内部阈值设定，通常不需要精确调整
    clf_ecod.fit(X_point)
    S1 = clf_ecod.decision_scores_

    # --- 3. 阶段 2: Isolation Forest (局部隔离/情境异常) ---
    # IF 作用在展平后的窗口数据上，用于检测窗口的隔离程度
    clf_iforest = IForest(contamination=0.01, random_state=42)
    clf_iforest.fit(X_flat)
    S2 = clf_iforest.decision_scores_

    # --- 4. 阶段 3: LSTM-AutoEncoder (集合异常) ---
    n_samples, win_len, n_dims = X_windowed.shape
    ae_model = build_lstm_autoencoder(win_len, n_dims)
    
    # 假设 X_windowed 是训练/测试集，实际需分离训练集
    # 实际应用中需要使用正常数据训练 AE 模型
    # ae_model.fit(X_windowed, X_windowed, epochs=50, batch_size=32, verbose=0, callbacks=[EarlyStopping()])
    
    # 模拟训练后的预测
    S3 = get_reconstruction_scores(ae_model, X_windowed)

    # --- 5. 阶段 4: 分数融合 ---
    # 确保所有分数维度相同，并进行 Min-Max 归一化 (PyOD内置工具)
    S1_norm = min_max_scaler(S1)
    S2_norm = min_max_scaler(S2)
    S3_norm = min_max_scaler(S3)

    # 加权融合
    S_final = (ALPHA * S1_norm) + (BETA * S2_norm) + (GAMMA * S3_norm)
    
    return S_final, S1_norm, S2_norm, S3_norm