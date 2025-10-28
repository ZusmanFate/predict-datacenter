"""
数据预处理模块
处理缺失值、异常值、数据转换等
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """数据预处理器"""
    
    def __init__(self):
        """初始化数据预处理器"""
        self.scaler = None
        self.feature_cols = None
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'forward_fill',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据框
            method: 处理方法 ('forward_fill', 'backward_fill', 'interpolate', 'mean', 'median', 'drop')
            columns: 要处理的列（默认处理所有列）
            
        Returns:
            处理后的数据框
        """
        df = df.copy()
        target_cols = columns or df.columns.tolist()
        
        missing_count = df[target_cols].isnull().sum().sum()
        if missing_count == 0:
            logger.info("数据中无缺失值")
            return df
        
        logger.info(f"检测到 {missing_count} 个缺失值，使用 {method} 方法处理")
        
        if method == 'forward_fill':
            df[target_cols] = df[target_cols].fillna(method='ffill')
        elif method == 'backward_fill':
            df[target_cols] = df[target_cols].fillna(method='bfill')
        elif method == 'interpolate':
            df[target_cols] = df[target_cols].interpolate(method='linear')
        elif method == 'mean':
            df[target_cols] = df[target_cols].fillna(df[target_cols].mean())
        elif method == 'median':
            df[target_cols] = df[target_cols].fillna(df[target_cols].median())
        elif method == 'drop':
            df = df.dropna(subset=target_cols)
        else:
            raise ValueError(f"不支持的缺失值处理方法: {method}")
        
        # 处理仍然存在的缺失值（例如第一行的前向填充）
        remaining_missing = df[target_cols].isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"仍有 {remaining_missing} 个缺失值，使用0填充")
            df[target_cols] = df[target_cols].fillna(0)
        
        logger.info("缺失值处理完成")
        return df
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            df: 数据框
            column: 列名
            method: 方法 ('iqr', 'zscore', 'clip')
            threshold: 阈值（IQR倍数或Z-score阈值）
            
        Returns:
            处理后的数据框
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            logger.info(f"检测到 {outliers} 个异常值 (IQR method)")
            
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            z_scores = np.abs((df[column] - mean) / std)
            
            outliers = (z_scores > threshold).sum()
            logger.info(f"检测到 {outliers} 个异常值 (Z-score method)")
            
            df.loc[z_scores > threshold, column] = mean
        
        elif method == 'clip':
            lower_bound = df[column].quantile(0.01)
            upper_bound = df[column].quantile(0.99)
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        else:
            raise ValueError(f"不支持的异常值处理方法: {method}")
        
        return df
    
    def create_time_series_dataset(
        self,
        df: pd.DataFrame,
        drug_id: str,
        hospital_id: str,
        date_column: str = 'date',
        target_column: str = 'sales_quantity'
    ) -> pd.DataFrame:
        """
        为单个药品-医院组合创建时间序列数据集
        
        Args:
            df: 原始数据框
            drug_id: 药品ID
            hospital_id: 医院ID
            date_column: 日期列名
            target_column: 目标列名
            
        Returns:
            时间序列数据集
        """
        # 筛选特定药品和医院
        mask = (df['drug_id'] == drug_id) & (df['hospital_id'] == hospital_id)
        ts_df = df[mask].copy()
        
        if len(ts_df) == 0:
            logger.warning(f"未找到 drug_id={drug_id}, hospital_id={hospital_id} 的数据")
            return pd.DataFrame()
        
        # 按日期排序
        ts_df = ts_df.sort_values(date_column)
        
        # 确保日期连续性
        ts_df = self._ensure_date_continuity(ts_df, date_column, target_column)
        
        logger.info(f"创建时间序列数据集: drug_id={drug_id}, hospital_id={hospital_id}, 共 {len(ts_df)} 条记录")
        return ts_df
    
    def _ensure_date_continuity(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str
    ) -> pd.DataFrame:
        """
        确保日期连续性，填充缺失的日期
        
        Args:
            df: 数据框
            date_column: 日期列名
            target_column: 目标列名
            
        Returns:
            日期连续的数据框
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # 创建完整的日期范围
        date_range = pd.date_range(
            start=df[date_column].min(),
            end=df[date_column].max(),
            freq='D'
        )
        
        # 重新索引
        df = df.set_index(date_column)
        df = df.reindex(date_range)
        
        # 填充缺失的目标值（前向填充 + 后向填充）
        df[target_column] = df[target_column].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 重置索引
        df = df.reset_index()
        df = df.rename(columns={'index': date_column})
        
        return df
    
    def aggregate_by_frequency(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        target_column: str = 'sales_quantity',
        freq: str = 'D',
        agg_func: str = 'sum'
    ) -> pd.DataFrame:
        """
        按频率聚合数据
        
        Args:
            df: 数据框
            date_column: 日期列名
            target_column: 目标列名
            freq: 频率 ('D'=日, 'W'=周, 'M'=月, 'Q'=季度)
            agg_func: 聚合函数 ('sum', 'mean', 'median', 'max', 'min')
            
        Returns:
            聚合后的数据框
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        
        if agg_func == 'sum':
            result = df[target_column].resample(freq).sum()
        elif agg_func == 'mean':
            result = df[target_column].resample(freq).mean()
        elif agg_func == 'median':
            result = df[target_column].resample(freq).median()
        elif agg_func == 'max':
            result = df[target_column].resample(freq).max()
        elif agg_func == 'min':
            result = df[target_column].resample(freq).min()
        else:
            raise ValueError(f"不支持的聚合函数: {agg_func}")
        
        result = result.reset_index()
        logger.info(f"数据聚合完成: freq={freq}, agg_func={agg_func}, 结果数量={len(result)}")
        return result
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        归一化数据
        
        Args:
            df: 数据框
            columns: 要归一化的列
            method: 归一化方法 ('minmax', 'zscore')
            
        Returns:
            归一化后的数据框
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        df = df.copy()
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscore':
            scaler = StandardScaler()
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        df[columns] = scaler.fit_transform(df[columns])
        self.scaler = scaler
        
        logger.info(f"数据归一化完成: method={method}, columns={columns}")
        return df
    
    def split_sequences(
        self,
        df: pd.DataFrame,
        target_column: str,
        n_steps_in: int = 30,
        n_steps_out: int = 1
    ) -> tuple:
        """
        为深度学习模型创建序列数据
        
        Args:
            df: 数据框
            target_column: 目标列名
            n_steps_in: 输入序列长度
            n_steps_out: 输出序列长度
            
        Returns:
            (X, y) 序列数据
        """
        sequences = df[target_column].values
        X, y = [], []
        
        for i in range(len(sequences)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            
            if out_end_ix > len(sequences):
                break
            
            seq_x = sequences[i:end_ix]
            seq_y = sequences[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"序列数据创建完成: X shape={X.shape}, y shape={y.shape}")
        return X, y
