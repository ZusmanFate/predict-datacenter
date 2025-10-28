"""
基础模型类
定义所有预测模型的统一接口
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """基础预测模型抽象类"""
    
    def __init__(self, model_name: str = "base_model"):
        """
        初始化模型
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
    
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_valid: 验证特征
            y_valid: 验证目标
            **kwargs: 其他参数
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测值
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> 'BaseModel':
        """
        加载模型
        
        Args:
            filepath: 模型路径
            
        Returns:
            self
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            参数字典
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """
        设置模型参数
        
        Args:
            **params: 参数
            
        Returns:
            self
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        获取特征重要性
        
        Returns:
            特征重要性 DataFrame
        """
        if not self.is_fitted:
            logger.warning("模型尚未训练")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            return importance_df.sort_values('importance', ascending=False)
        
        logger.warning(f"{self.model_name} 不支持特征重要性")
        return None
    
    def __str__(self) -> str:
        return f"{self.model_name}(fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        return self.__str__()


class TimeSeriesModel(BaseModel):
    """时间序列模型基类"""
    
    def __init__(self, model_name: str = "timeseries_model"):
        super().__init__(model_name)
        self.forecast_horizon = 1
    
    def forecast(
        self,
        periods: int,
        exog: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        预测未来值
        
        Args:
            periods: 预测期数
            exog: 外生变量
            
        Returns:
            预测值
        """
        raise NotImplementedError("子类需要实现 forecast 方法")
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        horizon: int,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        时间序列交叉验证
        
        Args:
            df: 数据框
            horizon: 预测期
            n_splits: 折数
            
        Returns:
            评估指标
        """
        raise NotImplementedError("子类需要实现 cross_validate 方法")
