"""
辅助函数模块
提供通用的工具函数
"""
import yaml
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def load_config(config_path: str) -> Dict:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_pickle(obj: Any, filepath: str) -> None:
    """
    保存对象为 pickle 文件
    
    Args:
        obj: 要保存的对象
        filepath: 文件路径
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    加载 pickle 文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的对象
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_model(model: Any, filepath: str, use_joblib: bool = True) -> None:
    """
    保存模型
    
    Args:
        model: 模型对象
        filepath: 保存路径
        use_joblib: 是否使用 joblib（推荐用于大模型）
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    if use_joblib:
        joblib.dump(model, filepath)
    else:
        save_pickle(model, filepath)


def load_model(filepath: str, use_joblib: bool = True) -> Any:
    """
    加载模型
    
    Args:
        filepath: 模型路径
        use_joblib: 是否使用 joblib
        
    Returns:
        模型对象
    """
    if use_joblib:
        return joblib.load(filepath)
    else:
        return load_pickle(filepath)


def get_date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    生成日期范围
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
    Returns:
        日期索引
    """
    return pd.date_range(start=start_date, end=end_date, freq='D')


def create_directory(path: str) -> Path:
    """
    创建目录（如果不存在）
    
    Args:
        path: 目录路径
        
    Returns:
        Path 对象
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算预测指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        指标字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # 避免除零错误
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    # MAPE - 避免除零
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['mape'] = mape
    else:
        metrics['mape'] = np.inf
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    格式化指标输出
    
    Args:
        metrics: 指标字典
        
    Returns:
        格式化的字符串
    """
    formatted = []
    for key, value in metrics.items():
        if value == np.inf:
            formatted.append(f"{key.upper()}: Inf")
        else:
            formatted.append(f"{key.upper()}: {value:.4f}")
    return " | ".join(formatted)


def split_train_test(
    df: pd.DataFrame,
    date_column: str = 'date',
    test_size: float = 0.2
) -> tuple:
    """
    按时间顺序拆分训练集和测试集
    
    Args:
        df: 数据框
        date_column: 日期列名
        test_size: 测试集比例
        
    Returns:
        (train_df, test_df)
    """
    df = df.sort_values(date_column)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def add_time_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    添加时间特征
    
    Args:
        df: 数据框
        date_column: 日期列名
        
    Returns:
        添加特征后的数据框
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    
    return df


def get_timestamp() -> str:
    """
    获取当前时间戳字符串
    
    Returns:
        时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir_exists(filepath: str) -> None:
    """
    确保文件所在目录存在
    
    Args:
        filepath: 文件路径
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
