"""
Prophet 模型实现
用于时间序列预测（适合快速基线建模）
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base import TimeSeriesModel
from src.utils.logger import get_logger
from src.utils.helpers import load_config, save_pickle, load_pickle

logger = get_logger(__name__)


class ProphetModel(TimeSeriesModel):
    """Prophet 预测模型"""
    
    def __init__(self, config_path: str = "config/model_config.yaml", **kwargs):
        """
        初始化 Prophet 模型
        
        Args:
            config_path: 模型配置文件路径
            **kwargs: 额外的模型参数
        """
        super().__init__("Prophet")
        
        # 延迟导入 Prophet（因为可能需要安装）
        try:
            from prophet import Prophet
            self.Prophet = Prophet
        except ImportError:
            logger.error("Prophet 未安装，请运行: pip install prophet")
            raise
        
        # 加载配置
        config = load_config(config_path)
        prophet_config = config.get('prophet', {})
        
        # 模型参数
        self.params = prophet_config.get('default_params', {})
        self.params.update(kwargs)
        
        self.model = None
        
        logger.info("Prophet 模型初始化完成")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        date_column: str = 'date',
        **kwargs
    ) -> 'ProphetModel':
        """
        训练模型
        
        Args:
            X_train: 训练特征（必须包含日期列）
            y_train: 训练目标
            X_valid: 验证特征（Prophet不使用）
            y_valid: 验证目标（Prophet不使用）
            date_column: 日期列名
            **kwargs: 其他参数
            
        Returns:
            self
        """
        logger.info(f"开始训练 Prophet 模型，训练集大小: {len(X_train)}")
        
        # Prophet 需要 'ds' 和 'y' 列
        if date_column not in X_train.columns:
            raise ValueError(f"训练数据中未找到日期列: {date_column}")
        
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(X_train[date_column]),
            'y': y_train.values
        })
        
        # 创建并训练模型
        self.model = self.Prophet(**self.params)
        
        # 添加自定义季节性（如果配置了）
        config = load_config("config/model_config.yaml")
        custom_seasonality = config.get('prophet', {}).get('custom_seasonality', [])
        for season in custom_seasonality:
            self.model.add_seasonality(
                name=season['name'],
                period=season['period'],
                fourier_order=season['fourier_order']
            )
        
        self.model.fit(df_prophet)
        self.is_fitted = True
        
        logger.info("Prophet 模型训练完成")
        return self
    
    def predict(self, X: pd.DataFrame, date_column: str = 'date') -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据（必须包含日期列）
            date_column: 日期列名
            
        Returns:
            预测值
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit 方法")
        
        if date_column not in X.columns:
            raise ValueError(f"预测数据中未找到日期列: {date_column}")
        
        # 准备预测数据
        future = pd.DataFrame({
            'ds': pd.to_datetime(X[date_column])
        })
        
        # 预测
        forecast = self.model.predict(future)
        predictions = forecast['yhat'].values
        
        return predictions
    
    def forecast(
        self,
        periods: int,
        freq: str = 'D',
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        预测未来值
        
        Args:
            periods: 预测期数
            freq: 频率 ('D'=日, 'W'=周, 'M'=月)
            include_history: 是否包含历史数据
            
        Returns:
            预测结果 DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 创建未来日期
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )
        
        # 预测
        forecast = self.model.predict(future)
        
        logger.info(f"预测 {periods} 期未来值")
        return forecast
    
    def save(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        save_pickle(self.model, filepath)
        
        logger.info(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str) -> 'ProphetModel':
        """
        加载模型
        
        Args:
            filepath: 模型路径
            
        Returns:
            self
        """
        self.model = load_pickle(filepath)
        self.is_fitted = True
        
        logger.info(f"模型已从 {filepath} 加载")
        return self
    
    def plot_forecast(self, forecast: pd.DataFrame):
        """
        绘制预测结果
        
        Args:
            forecast: 预测结果 DataFrame
        """
        from prophet.plot import plot_plotly, plot_components_plotly
        import plotly.offline as py
        
        # 绘制预测图
        fig1 = plot_plotly(self.model, forecast)
        py.plot(fig1, filename='prophet_forecast.html')
        
        # 绘制组件图（趋势、季节性等）
        fig2 = plot_components_plotly(self.model, forecast)
        py.plot(fig2, filename='prophet_components.html')
        
        logger.info("预测图已生成: prophet_forecast.html, prophet_components.html")
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        horizon: str = '30 days',
        initial: str = '365 days',
        period: str = '90 days'
    ) -> pd.DataFrame:
        """
        时间序列交叉验证
        
        Args:
            df: 数据框（必须有 'ds' 和 'y' 列）
            horizon: 预测期
            initial: 初始训练集大小
            period: 验证集间隔
            
        Returns:
            交叉验证结果
        """
        from prophet.diagnostics import cross_validation, performance_metrics
        
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 交叉验证
        df_cv = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        # 计算性能指标
        df_metrics = performance_metrics(df_cv)
        
        logger.info(f"交叉验证完成，平均 MAPE: {df_metrics['mape'].mean():.4f}")
        
        return df_cv, df_metrics
