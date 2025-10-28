"""
超参数优化模块
使用 Optuna 进行自动化超参数搜索
"""
import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lgb_model import LightGBMModel
from src.utils.logger import get_logger
from src.utils.helpers import calculate_metrics, load_config

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(
        self,
        model_type: str = 'lightgbm',
        n_trials: int = 100,
        direction: str = 'minimize',
        metric: str = 'rmse'
    ):
        """
        初始化优化器
        
        Args:
            model_type: 模型类型
            n_trials: 试验次数
            direction: 优化方向 ('minimize' 或 'maximize')
            metric: 优化指标
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.direction = direction
        self.metric = metric
        self.study = None
        self.best_params = None
        
        logger.info(f"超参数优化器初始化: {model_type}, {n_trials} 次试验, 优化 {metric}")
    
    def optimize_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        param_space: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        优化 LightGBM 超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_valid: 验证特征
            y_valid: 验证目标
            param_space: 参数搜索空间（可选）
            
        Returns:
            最佳参数
        """
        logger.info("开始 LightGBM 超参数优化...")
        
        # 加载默认参数空间
        if param_space is None:
            config = load_config("config/model_config.yaml")
            param_space = config.get('lightgbm', {}).get('optuna_space', {})
        
        def objective(trial):
            """Optuna 目标函数"""
            # 建议参数
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42
            }
            
            # 从搜索空间采样
            if 'num_leaves' in param_space:
                params['num_leaves'] = trial.suggest_int(
                    'num_leaves',
                    param_space['num_leaves'][0],
                    param_space['num_leaves'][1]
                )
            
            if 'learning_rate' in param_space:
                params['learning_rate'] = trial.suggest_float(
                    'learning_rate',
                    param_space['learning_rate'][0],
                    param_space['learning_rate'][1],
                    log=True
                )
            
            if 'feature_fraction' in param_space:
                params['feature_fraction'] = trial.suggest_float(
                    'feature_fraction',
                    param_space['feature_fraction'][0],
                    param_space['feature_fraction'][1]
                )
            
            if 'bagging_fraction' in param_space:
                params['bagging_fraction'] = trial.suggest_float(
                    'bagging_fraction',
                    param_space['bagging_fraction'][0],
                    param_space['bagging_fraction'][1]
                )
            
            if 'max_depth' in param_space:
                params['max_depth'] = trial.suggest_int(
                    'max_depth',
                    param_space['max_depth'][0],
                    param_space['max_depth'][1]
                )
            
            if 'min_data_in_leaf' in param_space:
                params['min_data_in_leaf'] = trial.suggest_int(
                    'min_data_in_leaf',
                    param_space['min_data_in_leaf'][0],
                    param_space['min_data_in_leaf'][1]
                )
            
            params['bagging_freq'] = 5
            
            # 训练模型
            model = LightGBMModel(**params)
            model.fit(X_train, y_train, X_valid, y_valid)
            
            # 预测和评估
            y_pred = model.predict(X_valid)
            metrics = calculate_metrics(y_valid, y_pred)
            
            return metrics[self.metric]
        
        # 创建研究
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # 获取最佳参数
        self.best_params = self.study.best_params
        
        logger.info(f"优化完成！最佳 {self.metric}: {self.study.best_value:.4f}")
        logger.info(f"最佳参数: {self.best_params}")
        
        return self.best_params
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        绘制优化历史
        
        Args:
            save_path: 保存路径
        """
        if self.study is None:
            logger.warning("尚未进行优化")
            return
        
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 优化历史
        plot_optimization_history(self.study, ax=axes[0])
        axes[0].set_title('优化历史', fontsize=14, fontweight='bold')
        
        # 参数重要性
        plot_param_importances(self.study, ax=axes[1])
        axes[1].set_title('参数重要性', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"优化历史图已保存到: {save_path}")
        
        plt.show()
    
    def get_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ):
        """
        使用最佳参数训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_valid: 验证特征
            y_valid: 验证目标
            
        Returns:
            训练好的模型
        """
        if self.best_params is None:
            raise ValueError("尚未进行优化，请先调用 optimize 方法")
        
        logger.info("使用最佳参数训练模型...")
        
        if self.model_type == 'lightgbm':
            model = LightGBMModel(**self.best_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        model.fit(X_train, y_train, X_valid, y_valid)
        
        logger.info("模型训练完成")
        return model
    
    def save_study(self, filepath: str):
        """
        保存优化研究
        
        Args:
            filepath: 保存路径
        """
        if self.study is None:
            logger.warning("尚未进行优化")
            return
        
        import joblib
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.study, filepath)
        logger.info(f"优化研究已保存到: {filepath}")
    
    def load_study(self, filepath: str):
        """
        加载优化研究
        
        Args:
            filepath: 文件路径
        """
        import joblib
        self.study = joblib.load(filepath)
        self.best_params = self.study.best_params
        logger.info(f"优化研究已从 {filepath} 加载")
