import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.copod import COPOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD

# 导入深度学习模型
try:
    from pyod.models.auto_encoder import AutoEncoder
    AUTOENCODER_AVAILABLE = True
except ImportError:
    AUTOENCODER_AVAILABLE = False
# try:
#     from pyod.models.deep_svdd import DeepSVDD
# except ImportError:
#     print("DeepSVDD依赖缺失, 跳过DeepSVDD模型")
#     AUTOENCODER_AVAILABLE = False

import time
from datetime import datetime, timedelta


def generate_financial_operational_data(n_samples=10000, n_features=15, contamination=0.05):

    # 生成金融运营数据 + OpenTelemetry 服务器链路监控数据

    # 指标说明（15维特征）：
    # 传统运营指标（Traditional Operational Metrics）:
    # - API 响应时间(ms) -> API Response Time
    # - 批处理延时(ms) -> Batch Processing Delay
    # - 错误率(%) -> Error Rate
    # - 吞吐量(records/min) -> Throughput
    # - CPU 使用率(%) -> CPU Usage
    # - 内存使用率(%) -> Memory Usage
    # - 数据质量评分(0-100) -> Data Quality Score
    # - 并发连接数 -> Concurrent Connections

    # OpenTelemetry 服务指标（Server-side Metrics）:
    # - HTTP 平均响应时间(ms) -> HTTP Request Latency P95
    # - CPU 峰值 -> CPU Peak Usage
    # - 服务调用持续时间 -> Service Trace Duration
    # - 队列时延(ms) -> Queue latency
    # - Span Duration(ms) -> Distributed Trace Span Duration
    # - 每秒查询数(QPS) -> Service Queries Per Second
    # - HTTP 错误率(%) -> HTTP 5xx Error Rate

    np.random.seed(42)

    # 正常数据
    n_normal = int(n_samples * (1 - contamination))
    n_anomaly = n_samples - n_normal

    # 正常高斯数据 - 多维正态分布
    normal_data = np.random.randn(n_normal, n_features)

    # 传统运营指标和服务器指标逐项调整分布
    # 特征维度映射（0-7）：
    # Feature 0: API 响应时间(50-200ms)
    normal_data[:, 0] = np.abs(normal_data[:, 0] * 30 + 125)

    # Feature 1: 批处理延时 (1-10s)
    normal_data[:, 1] = np.abs(normal_data[:, 1] * 2 + 5)

    # Feature 2: 错误率 (0-2%)
    normal_data[:, 2] = np.abs(normal_data[:, 2] * 0.5 + 1)

    # Feature 3: 吞吐量 (800-1200 records/min)
    normal_data[:, 3] = np.abs(normal_data[:, 3] * 100 + 1000)

    # Feature 4: CPU 使用率 (30-70%)
    normal_data[:, 4] = np.abs(normal_data[:, 4] * 10 + 50)

    # Feature 5: 内存使用率 (40-80%)
    normal_data[:, 5] = np.abs(normal_data[:, 5] * 10 + 60)

    # Feature 6: 数据质量分 (70-100)
    normal_data[:, 6] = np.abs(normal_data[:, 6] * 3 + 92)

    # Feature 7: 并发连接数 (50-200)
    normal_data[:, 7] = np.abs(normal_data[:, 7] * 25 + 100)

    # OpenTelemetry 服务器指标（8-14）
    # Feature 8: HTTP 响应 P95 (80-200ms)
    normal_data[:, 8] = np.abs(normal_data[:, 8] * 35 + 125)

    # Feature 9: CPU 峰值(85-98%)
    normal_data[:, 9] = np.abs(normal_data[:, 9] * 10 + 27.5)

    # Feature 10: 服务队列深度(5-20)
    normal_data[:, 10] = np.abs(normal_data[:, 10] * 3 + 9.6)

    # Feature 11: 分布式追踪耗时(100-180ms)
    normal_data[:, 11] = np.abs(normal_data[:, 11] * 20 + 65)

    # Feature 12: Trace Span Duration (20-150ms)
    normal_data[:, 12] = np.abs(normal_data[:, 12] * 30 + 85)

    # Feature 13: QPS 每秒调用量(800-1500 queries/sec)
    normal_data[:, 13] = np.abs(normal_data[:, 13] * 350 + 1260)

    # Feature 14: HTTP 5xx 错误率 (0-3%)
    normal_data[:, 14] = np.abs(normal_data[:, 14] * 0.1 + 0.25)


    # 生成异常数据（多源）
    # 极端异常：完全随机噪声异常（30%）
    n_extreme = int(n_anomaly * 0.3)
    extreme_anomaly = np.random.randn(n_extreme, n_features) * 3 + 5
    
    # 类型2 局部异常：部分指标偏移异常（30%）
    n_local = int(n_anomaly * 0.3)
    local_anomaly = np.random.randn(n_local, n_features)
    # 随机选择2～3个特征为异常
    for i in range(n_local):
        anomaly_features = np.random.choice(n_features, size=np.random.randint(2, 4), replace=False)
        local_anomaly[i, anomaly_features] *= 4

    # 类别3：服务端性能劣化异常（20%） - 模拟服务降级
    n_service = int(n_anomaly * 0.2)
    service_anomaly = np.random.randn(n_service, n_features)

    # 前8个特征正常
    for j in range(8):
        if j == 0:
            service_anomaly[:, j] = service_anomaly[:, j] * 30 + 125
        elif j == 1:
            service_anomaly[:, j] = np.abs(service_anomaly[:, j] * 2 + 5)
        elif j == 2:
            service_anomaly[:, j] = np.abs(service_anomaly[:, j] * 0.5 + 1)
        elif j == 3:
            service_anomaly[:, j] = service_anomaly[:, j] * 100 + 1000
        elif j == 4:
            service_anomaly[:, j] = service_anomaly[:, j] * 10 + 50
        elif j == 5:
            service_anomaly[:, j] = service_anomaly[:, j] * 10 + 60
        elif j == 6:
            service_anomaly[:, j] = service_anomaly[:, j] * 3 + 92
        elif j == 7:
            service_anomaly[:, j] = service_anomaly[:, j] * 25 + 100

    # OpenTelemetry 服务侧指标异常（8-14）
    service_anomaly[:, 8]  = np.abs(service_anomaly[:, 8]  * 100 + 500)   # HTTP P95延迟高
    service_anomaly[:, 9]  = np.abs(service_anomaly[:, 9]  * 15  + 90)    # 数据峰值高
    service_anomaly[:, 10] = np.abs(service_anomaly[:, 10] * 25  + 50)    # 队列时延升高
    service_anomaly[:, 11] = np.abs(service_anomaly[:, 11] * 30  + 100)   # Trace 耗时变高
    service_anomaly[:, 12] = np.abs(service_anomaly[:, 12] * 40  + 150)   # Span Duration长
    service_anomaly[:, 13] = np.abs(service_anomaly[:, 13] * 300 + 900)   # QPS低（服务器降级）
    service_anomaly[:, 14] = np.abs(service_anomaly[:, 14] * 2   + 5)     # 5xx错误率高

    # 类别4：系统级故障异常（20%）
    n_pattern = n_anomaly - n_extreme - n_local - n_service
    pattern_anomaly = np.random.randn(n_pattern, n_features)
    pattern_anomaly = pattern_anomaly * 2 + np.array([
        150, 5, 3, 900, 55, 60, 98, 100,
        800, 500, 30, 100, 200, 1000, 10   # OpenTelemetry服务端指标异常
    ])

    # 合并数据
    X = np.vstack([normal_data, extreme_anomaly, local_anomaly, service_anomaly, pattern_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])

    # 打乱数据
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    # 创建特征名称（中文对应）
    feature_names = [
        # 传统运营指标 (Traditional Operational Metrics)
        'API_Response_Time_ms',      # API响应时间(毫秒)
        'Processing_Delay_sec',      # 批处理延迟(秒)
        'Error_Rate_pct',            # 错误率(%)
        'Throughput_records_min',    # 吞吐量(记录/分钟)
        'CPU_Usage_pct',             # CPU使用率(%)
        'Memory_Usage_pct',          # 内存使用率(%)
        'Data_Quality_Score',        # 数据质量评分
        'Concurrent_Connections',    # 并发连接数

        # OpenTelemetry 服务侧指标 (Server-side Metrics)
        'HTTP_P95_Latency_ms',       # HTTP请求P95延迟(毫秒)
        'Peak_CPU_pct',              # CPU峰值(%)
        'Queue_Duration_ms',         # 数据队列时延(毫秒)
        'Cache_Hit_Ratio_pct',       # 缓存命中率(%)
        'Span_Duration_ms',          # 分布式追踪Span时长(毫秒)
        'Service_QPS',               # 服务查询率 (QPS)
        'HTTP_5xx_Error_Rate_pct'    # HTTP 5xx错误率(%)
    ]

    return X, y, feature_names


def generate_time_series_features(X, timestamps=None):
    """
    生成时序特征 - 针对大规模时序金融数据
    Time-series feature engineering for large-scale financial data
    """
    n_samples = X.shape[0]

    # 如果没有提供时间戳，生成模拟时间戳 (每分钟一个数据点)
    if timestamps is None:
        start_time = datetime(2025, 1, 1)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]

    # 提取时间特征
    time_features = pd.DataFrame({
        'hour': [t.hour for t in timestamps],
        'day_of_week': [t.weekday() for t in timestamps],
        'day_of_month': [t.day for t in timestamps],
        'is_weekend': [1 if t.weekday() >= 5 else 0 for t in timestamps],
        'is_business_hour': [1 if 9 <= t.hour < 18 else 0 for t in timestamps],
    })

    # 周期性编码 (sin/cos 变换避免边界不连续)
    time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24)
    time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24)
    time_features['day_sin'] = np.sin(2 * np.pi * time_features['day_of_week'] / 7)
    time_features['day_cos'] = np.cos(2 * np.pi * time_features['day_of_week'] / 7)

    # 滑动窗口统计特征 (捕获时序依赖)
    window_sizes = [5, 10, 30]  # 5分钟、10分钟、30分钟窗口
    X_df = pd.DataFrame(X)

    for col in range(X.shape[1]):
        for window in window_sizes:
            # 移动平均
            time_features[f'feature_{col}_ma_{window}'] = X_df[col].rolling(window=window, min_periods=1).mean()
            # 移动标准差
            time_features[f'feature_{col}_std_{window}'] = X_df[col].rolling(window=window, min_periods=1).std().fillna(0)

    # 合并原始特征和时序特征
    X_combined = np.hstack([X, time_features.values])

    return X_combined, time_features.columns.tolist()


def evaluate_time_series_performance(model, X_train, X_test, y_test, timestamps_test=None):
    """
    时序数据专用评估指标
    Time-series specific evaluation metrics
    """
    # 基础性能指标
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)

    metrics = {
        'auc': roc_auc_score(y_test, y_scores),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    # 时序专用指标: 检测延迟 (Detection Delay)
    # 计算从异常开始到被检测出的平均延迟
    if timestamps_test is not None and len(timestamps_test) == len(y_test):
        anomaly_indices = np.where(y_test == 1)[0]
        detected_indices = np.where(y_pred == 1)[0]

        # 简化计算：统计有多少异常在合理时间窗口内被检测
        metrics['detection_window_5min'] = 0
        for anomaly_idx in anomaly_indices:
            # 检查后续5个时间点内是否检测到
            window = detected_indices[(detected_indices >= anomaly_idx) & (detected_indices < anomaly_idx + 5)]
            if len(window) > 0:
                metrics['detection_window_5min'] += 1

        if len(anomaly_indices) > 0:
            metrics['detection_window_5min'] /= len(anomaly_indices)

    # 误报率 (False Positive Rate) - 时序数据中的关键指标
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return metrics


def compare_algorithms(X_train, X_test, y_test, contamination=0.05, enable_time_series_features=False):
    """
    比较多个异常检测算法 (针对大规模时序金融数据优化)
    参数:
    - enable_time_series_features: 是否启用时序特征工程 (会增加特征维度)
    """
    print(f"数据规模: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    print(f"时序特征工程: {'启用' if enable_time_series_features else '禁用'}")
    # 定义算法 - 针对大规模时序金融数据优化
    # 优先考虑: 1) 可扩展性 2) 训练速度 3) 在线学习能力
    algorithms = {
        # ☆ 推荐用于大规模时序数据的算法
        'Isolation Forest': IForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,  # 从200降至100以加速
            max_samples='auto',  # 自动采样，适合大数据
            n_jobs=-1 # 并行训练
        ),
        'ECOD': ECOD(
            contamination=contamination,
            n_jobs=-1 # 并行计算
        ),
        'HBOS': HBOS(
            contamination=contamination,
            n_bins=15,  # 从20降至15以加速
            tol=0.5  # 容差设置
        ),
        'COPOD': COPOD(
            contamination=contamination,
            n_jobs=-1
        ),
        # ◆ 中等推荐 - 性能良好但可能在超大规模数据上较慢
        'PCA': PCA(
            contamination=contamination,
            n_components=min(8, X_train.shape[1] // 2)  # 动态设置主成分数
        ),
        # ▼ 不推荐用于大规模数据 - 仅作对比
        'KNN': KNN(contamination=contamination, n_neighbors=10, n_jobs=-1),
        'LOF': LOF(contamination=contamination, n_neighbors=20, n_jobs=-1),
    }

    # 可选: 添加深度学习算法 (适合离线批处理)
    if AUTOENCODER_AVAILABLE:
        try:
            algorithms['AutoEncoder'] = AutoEncoder(
                contamination=contamination,
                hidden_neuron_list=[X_train.shape[1], max(10, X_train.shape[1]//2), max(10, X_train.shape[1]//2), X_train.shape[1]],
                epoch_num=30,  # 从50降至30以加速
                batch_size=64,  # 从32增至64以提升吞吐量
                verbose=0,
                random_state=42
            )
            print("  ✅ AutoEncoder 已启用 (适合离线分析) ")
        except Exception as e:
            print(f"  ⚠️ AutoEncoder 初始化失败: {str(e)}")

    results = []
    print("\n开始算法比较 (针对大规模时序数据优化) ...")
    print("-" * 80)

    for name, model in algorithms.items():
        print(f"\n训练 {name}...")
        start_time = time.time()
        try:
            # 训练
            model.fit(X_train)
            # 预测
            y_pred = model.predict(X_test)
            y_scores = model.decision_function(X_test)
            # 计算指标
            training_time = time.time() - start_time
            # 预测时间
            start_pred = time.time()
            _ = model.predict(X_test[:100])
            prediction_time = (time.time() - start_pred) / 100 * 1000  # ms per sample

            auc = roc_auc_score(y_test, y_scores)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # 大数据特定指标: 吞吐量 (每秒处理样本数)
            throughput = len(X_test) / (time.time() - start_pred) if (time.time() - start_pred) > 0 else 0

            # 内存效率评估 (简化版)
            memory_friendly = "是" if name in ['Isolation Forest', 'ECOD', 'HBOS', 'COPOD'] else "否"
            online_learning = "支持" if name in ['HBOS'] else "不支持"
            results.append({
                'Algorithm': name,
                'AUC-ROC': auc,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Training Time (s)': training_time,
                'Prediction Time (ms)': prediction_time,
                'Throughput (samples/s)': throughput,
                'Memory Efficient': memory_friendly,
                'Scalability': '优秀' if name in ['Isolation Forest', 'ECOD'] else '一般' if name in ['HBOS', 'COPOD', 'PCA'] else '差'
            })
            print(f"  AUC: {auc:.4f}, F1: {f1:.4f}, 训练: {training_time:.2f}s, 吞吐量: {throughput:.0f} samples/s")

        except Exception as e:
            print(f"  ❌ 失败: {str(e)}")
            continue

    return pd.DataFrame(results)


def plot_comparison_results(results_df, output_path='algorithm_comparison.png'):
    """
    可视化比较结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 性能指标比较
    metrics = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    results_plot = results_df.set_index('Algorithm')[metrics]

    ax1 = axes[0, 0]
    results_plot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])

    # 2. AUC-ROC 排名
    ax2 = axes[0, 1]
    sorted_auc = results_df.sort_values('AUC-ROC', ascending=True)
    colors = plt.cm.RdYlGn(sorted_auc['AUC-ROC'].values)
    ax2.barh(sorted_auc['Algorithm'], sorted_auc['AUC-ROC'], color=colors)
    ax2.set_xlabel('AUC-ROC Score', fontsize=12)
    ax2.set_title('AUC-ROC Ranking', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # 3. 训练时间比较
    ax3 = axes[1, 0]
    sorted_time = results_df.sort_values('Training Time (s)')
    ax3.bar(range(len(sorted_time)), sorted_time['Training Time (s)'],
            color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(sorted_time)))
    ax3.set_xticklabels(sorted_time['Algorithm'], rotation=45, ha='right')
    ax3.set_ylabel('Training Time (seconds)', fontsize=12)
    ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Precision vs Recall
    ax4 = axes[1, 1]
    for idx, row in results_df.iterrows():
        ax4.scatter(row['Recall'], row['Precision'], s=200, alpha=0.6)
        ax4.annotate(row['Algorithm'],
                     (row['Recall'], row['Precision']),
                     fontsize=9, ha='center')
    ax4.set_xlabel('Recall', fontsize=12)
    ax4.set_ylabel('Precision', fontsize=12)
    ax4.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.set_xlim([0, 1.1])
    ax4.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 图表已保存: {output_path}")

    return fig
