"""
LightGBM 模型实现
用于时间序列回归预测
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base import BaseModel
from src.utils.logger import get_logger
from src.utils.helpers import load_config, save_model, load_model

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM 预测模型"""
    
    def __init__(self, config_path: str = "config/model_config.yaml", **kwargs):
        """
        初始化 LightGBM 模型
        
        Args:
            config_path: 模型配置文件路径
            **kwargs: 额外的模型参数（会覆盖配置文件）
        """
        super().__init__("LightGBM")
        
        # 加载配置
        config = load_config(config_path)
        lgb_config = config.get('lightgbm', {})
        
        # 模型参数
        self.params = lgb_config.get('default_params', {})
        self.params.update(kwargs)  # 用户参数覆盖
        
        # 训练参数
        self.training_params = lgb_config.get('training_params', {})
        
        self.model = None
        self.best_iteration = None
        self.best_score = None
        
        logger.info(f"LightGBM 模型初始化完成")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        categorical_features: Optional[list] = None,
        **kwargs
    ) -> 'LightGBMModel':
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_valid: 验证特征
            y_valid: 验证目标
            categorical_features: 类别特征列表
            **kwargs: 其他训练参数
            
        Returns:
            self
        """
        logger.info(f"开始训练 LightGBM 模型，训练集大小: {X_train.shape}")
        
        # 保存特征名
        self.feature_names = X_train.columns.tolist()
        
        # 创建数据集
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features
        )
        
        # 合并训练参数
        train_params = self.training_params.copy()
        train_params.update(kwargs)
        
        # 是否使用验证集
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(
                X_valid,
                label=y_valid,
                reference=train_data,
                categorical_feature=categorical_features
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')
            logger.info(f"使用验证集，大小: {X_valid.shape}")
        
        # 训练模型
        callbacks = [
            lgb.log_evaluation(period=train_params.get('verbose_eval', 50)),
            lgb.early_stopping(
                stopping_rounds=train_params.get('early_stopping_rounds', 50),
                first_metric_only=True,
                verbose=True
            ) if len(valid_sets) > 1 else None
        ]
        callbacks = [cb for cb in callbacks if cb is not None]
        
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=train_params.get('num_boost_round', 1000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.best_iteration = self.model.best_iteration
        self.best_score = self.model.best_score
        self.is_fitted = True
        
        logger.info(f"模型训练完成，最佳迭代: {self.best_iteration}, 最佳分数: {self.best_score}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测值
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit 方法")
        
        predictions = self.model.predict(
            X,
            num_iteration=self.best_iteration
        )
        
        return predictions
    
    def save(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存 LightGBM 模型
        self.model.save_model(filepath)
        
        # 保存元数据
        metadata = {
            'feature_names': self.feature_names,
            'best_iteration': self.best_iteration,
            'best_score': self.best_score,
            'params': self.params
        }
        
        import json
        metadata_path = filepath + '.meta.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str) -> 'LightGBMModel':
        """
        加载模型
        
        Args:
            filepath: 模型路径
            
        Returns:
            self
        """
        # 加载 LightGBM 模型
        self.model = lgb.Booster(model_file=filepath)
        
        # 加载元数据
        import json
        metadata_path = filepath + '.meta.json'
        if Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get('feature_names')
            self.best_iteration = metadata.get('best_iteration')
            self.best_score = metadata.get('best_score')
            self.params = metadata.get('params', {})
        
        self.is_fitted = True
        logger.info(f"模型已从 {filepath} 加载")
        return self
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 ('gain', 'split')
            
        Returns:
            特征重要性 DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_importance(
        self,
        importance_type: str = 'gain',
        max_num_features: int = 20,
        figsize: tuple = (10, 8)
    ):
        """
        绘制特征重要性图
        
        Args:
            importance_type: 重要性类型
            max_num_features: 最多显示的特征数
            figsize: 图形大小
        """
        import matplotlib.pyplot as plt
        
        lgb.plot_importance(
            self.model,
            importance_type=importance_type,
            max_num_features=max_num_features,
            figsize=figsize
        )
        plt.tight_layout()
        plt.show()
