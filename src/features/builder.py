"""
特征工程模块
构建时间序列特征：滞后特征、滚动统计特征、日期特征等
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)


class FeatureBuilder:
    """特征构建器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化特征构建器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.feature_config = self.config.get('features', {})
    
    def build_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'sales_quantity',
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        构建完整特征集
        
        Args:
            df: 数据框
            target_column: 目标列名
            date_column: 日期列名
            
        Returns:
            包含特征的数据框
        """
        logger.info("开始构建特征...")
        df = df.copy()
        
        # 1. 日期特征
        df = self.add_date_features(df, date_column)
        
        # 2. 滞后特征
        df = self.add_lag_features(df, target_column)
        
        # 3. 滚动窗口统计特征
        df = self.add_rolling_features(df, target_column)
        
        # 4. 差分特征
        df = self.add_diff_features(df, target_column)
        
        # 5. 统计特征
        df = self.add_statistical_features(df, target_column)
        
        # 删除因滞后产生的缺失值
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"删除缺失值: {initial_len} -> {len(df)} 条记录")
        
        logger.info(f"特征构建完成，共 {len(df.columns)} 个特征")
        return df
    
    def add_date_features(
        self,
        df: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        添加日期相关特征
        
        Args:
            df: 数据框
            date_column: 日期列名
            
        Returns:
            添加特征后的数据框
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        date_features = self.feature_config.get('date_features', [])
        
        for feature in date_features:
            if feature == 'year':
                df['year'] = df[date_column].dt.year
            elif feature == 'month':
                df['month'] = df[date_column].dt.month
            elif feature == 'day':
                df['day'] = df[date_column].dt.day
            elif feature == 'dayofweek':
                df['dayofweek'] = df[date_column].dt.dayofweek
            elif feature == 'quarter':
                df['quarter'] = df[date_column].dt.quarter
            elif feature == 'is_weekend':
                df['is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
            elif feature == 'is_month_start':
                df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
            elif feature == 'is_month_end':
                df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
            elif feature == 'is_quarter_start':
                df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
            elif feature == 'is_quarter_end':
                df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
        
        # 周期性编码（sin/cos）
        df['month_sin'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek / 7)
        
        logger.info(f"添加日期特征: {len(date_features)} 个基础特征 + 4 个周期性特征")
        return df
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        添加滞后特征
        
        Args:
            df: 数据框
            target_column: 目标列名
            lags: 滞后期列表
            
        Returns:
            添加特征后的数据框
        """
        df = df.copy()
        
        if lags is None:
            lags = self.feature_config.get('lag_features', [1, 2, 3, 7, 14, 30])
        
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        logger.info(f"添加滞后特征: {len(lags)} 个 (lags={lags})")
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        windows: Optional[List[int]] = None,
        statistics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        添加滚动窗口统计特征
        
        Args:
            df: 数据框
            target_column: 目标列名
            windows: 窗口大小列表
            statistics: 统计量列表
            
        Returns:
            添加特征后的数据框
        """
        df = df.copy()
        
        if windows is None:
            windows = self.feature_config.get('rolling_windows', [7, 14, 30, 90])
        
        if statistics is None:
            statistics = self.feature_config.get('statistics', ['mean', 'std', 'min', 'max'])
        
        for window in windows:
            for stat in statistics:
                feature_name = f'{target_column}_rolling_{window}_{stat}'
                
                if stat == 'mean':
                    df[feature_name] = df[target_column].rolling(window=window).mean()
                elif stat == 'std':
                    df[feature_name] = df[target_column].rolling(window=window).std()
                elif stat == 'min':
                    df[feature_name] = df[target_column].rolling(window=window).min()
                elif stat == 'max':
                    df[feature_name] = df[target_column].rolling(window=window).max()
                elif stat == 'median':
                    df[feature_name] = df[target_column].rolling(window=window).median()
        
        logger.info(f"添加滚动窗口特征: {len(windows)} 个窗口 × {len(statistics)} 个统计量")
        return df
    
    def add_diff_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        periods: List[int] = [1, 7]
    ) -> pd.DataFrame:
        """
        添加差分特征
        
        Args:
            df: 数据框
            target_column: 目标列名
            periods: 差分周期
            
        Returns:
            添加特征后的数据框
        """
        df = df.copy()
        
        for period in periods:
            df[f'{target_column}_diff_{period}'] = df[target_column].diff(period)
        
        logger.info(f"添加差分特征: {len(periods)} 个")
        return df
    
    def add_statistical_features(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> pd.DataFrame:
        """
        添加统计特征（扩展窗口）
        
        Args:
            df: 数据框
            target_column: 目标列名
            
        Returns:
            添加特征后的数据框
        """
        df = df.copy()
        
        # 累计统计
        df[f'{target_column}_cumsum'] = df[target_column].cumsum()
        df[f'{target_column}_cummean'] = df[target_column].expanding().mean()
        df[f'{target_column}_cumstd'] = df[target_column].expanding().std()
        
        logger.info("添加统计特征: 累计和、累计均值、累计标准差")
        return df
    
    def add_holiday_features(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        holidays: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        添加节假日特征
        
        Args:
            df: 数据框
            date_column: 日期列名
            holidays: 节假日列表 (YYYY-MM-DD)
            
        Returns:
            添加特征后的数据框
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        if holidays:
            holiday_dates = pd.to_datetime(holidays)
            df['is_holiday'] = df[date_column].isin(holiday_dates).astype(int)
            
            # 节前节后特征
            df['days_to_holiday'] = df[date_column].apply(
                lambda x: min([abs((x - h).days) for h in holiday_dates] + [999])
            )
            df['is_before_holiday'] = (df['days_to_holiday'] <= 3).astype(int)
            
            logger.info(f"添加节假日特征: {len(holidays)} 个节假日")
        else:
            df['is_holiday'] = 0
            df['days_to_holiday'] = 999
            df['is_before_holiday'] = 0
        
        return df
    
    def add_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[tuple]
    ) -> pd.DataFrame:
        """
        添加交互特征
        
        Args:
            df: 数据框
            feature_pairs: 特征对列表
            
        Returns:
            添加特征后的数据框
        """
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # 乘法交互
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # 除法交互（避免除零）
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-5)
        
        logger.info(f"添加交互特征: {len(feature_pairs)} 对")
        return df
    
    def get_feature_importance(
        self,
        model,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            
        Returns:
            特征重要性 DataFrame
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            logger.warning("模型不支持特征重要性")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"特征重要性计算完成，Top 5: {importance_df.head()['feature'].tolist()}")
        return importance_df
    
    def select_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        method: str = 'correlation',
        threshold: float = 0.01
    ) -> List[str]:
        """
        特征选择
        
        Args:
            df: 数据框
            target_column: 目标列名
            method: 选择方法 ('correlation', 'variance')
            threshold: 阈值
            
        Returns:
            选中的特征列表
        """
        feature_cols = [col for col in df.columns if col != target_column and col != 'date']
        
        if method == 'correlation':
            # 基于相关性选择
            corr = df[feature_cols + [target_column]].corr()[target_column].abs()
            selected = corr[corr > threshold].index.tolist()
            selected = [f for f in selected if f != target_column]
        
        elif method == 'variance':
            # 基于方差选择（去除低方差特征）
            variances = df[feature_cols].var()
            selected = variances[variances > threshold].index.tolist()
        
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        logger.info(f"特征选择完成: {len(feature_cols)} -> {len(selected)} ({method}, threshold={threshold})")
        return selected
