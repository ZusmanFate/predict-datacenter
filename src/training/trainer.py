"""
模型训练器
统一的模型训练接口
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base import BaseModel
from src.utils.logger import get_logger
from src.utils.helpers import calculate_metrics, format_metrics, split_train_test
import mlflow

logger = get_logger(__name__)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: BaseModel,
        experiment_name: str = "drug_sales_forecast",
        tracking_uri: str = "mlruns"
    ):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            experiment_name: MLflow 实验名称
            tracking_uri: MLflow 跟踪 URI
        """
        self.model = model
        self.experiment_name = experiment_name
        
        # 设置 MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"训练器初始化完成，模型: {model.model_name}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        log_mlflow: bool = True,
        run_name: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_valid: 验证特征
            y_valid: 验证目标
            log_mlflow: 是否记录到 MLflow
            run_name: MLflow 运行名称
            **kwargs: 其他训练参数
            
        Returns:
            训练好的模型
        """
        logger.info(f"开始训练 {self.model.model_name} 模型")
        
        if log_mlflow:
            with mlflow.start_run(run_name=run_name):
                # 记录参数
                mlflow.log_params(self.model.get_params())
                mlflow.log_param("train_size", len(X_train))
                if X_valid is not None:
                    mlflow.log_param("valid_size", len(X_valid))
                
                # 训练模型
                self.model.fit(X_train, y_train, X_valid, y_valid, **kwargs)
                
                # 评估训练集
                train_pred = self.model.predict(X_train)
                train_metrics = calculate_metrics(y_train, train_pred)
                for metric_name, metric_value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", metric_value)
                
                logger.info(f"训练集指标: {format_metrics(train_metrics)}")
                
                # 评估验证集
                if X_valid is not None and y_valid is not None:
                    valid_pred = self.model.predict(X_valid)
                    valid_metrics = calculate_metrics(y_valid, valid_pred)
                    for metric_name, metric_value in valid_metrics.items():
                        mlflow.log_metric(f"valid_{metric_name}", metric_value)
                    
                    logger.info(f"验证集指标: {format_metrics(valid_metrics)}")
                
                # 记录模型
                mlflow.sklearn.log_model(self.model, "model")
                
                logger.info("训练完成，已记录到 MLflow")
        else:
            # 不使用 MLflow
            self.model.fit(X_train, y_train, X_valid, y_valid, **kwargs)
            logger.info("训练完成")
        
        return self.model
    
    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2,
        log_mlflow: bool = True
    ) -> Dict[str, float]:
        """
        交叉验证训练
        
        Args:
            X: 特征
            y: 目标
            n_splits: 折数
            test_size: 测试集比例
            log_mlflow: 是否记录到 MLflow
            
        Returns:
            平均指标字典
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        logger.info(f"开始 {n_splits} 折交叉验证训练")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_metrics = []
        
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
            logger.info(f"训练折 {fold + 1}/{n_splits}")
            
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            # 训练
            self.model.fit(X_train, y_train, X_valid, y_valid)
            
            # 评估
            valid_pred = self.model.predict(X_valid)
            metrics = calculate_metrics(y_valid, valid_pred)
            all_metrics.append(metrics)
            
            logger.info(f"折 {fold + 1} 指标: {format_metrics(metrics)}")
        
        # 计算平均指标
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            avg_metrics[f"{metric_name}_mean"] = np.mean(values)
            avg_metrics[f"{metric_name}_std"] = np.std(values)
        
        logger.info(f"交叉验证平均指标: {format_metrics({k: v for k, v in avg_metrics.items() if 'mean' in k})}")
        
        if log_mlflow:
            with mlflow.start_run(run_name="cross_validation"):
                for metric_name, metric_value in avg_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
        
        return avg_metrics
    
    def train_on_full_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'sales_quantity',
        test_size: float = 0.2,
        log_mlflow: bool = True,
        **kwargs
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        在完整数据集上训练
        
        Args:
            df: 完整数据框
            target_column: 目标列名
            test_size: 测试集比例
            log_mlflow: 是否记录到 MLflow
            **kwargs: 其他训练参数
            
        Returns:
            (模型, 测试集指标)
        """
        # 分离特征和目标
        feature_cols = [col for col in df.columns if col not in [target_column, 'date']]
        X = df[feature_cols]
        y = df[target_column]
        
        # 划分训练集和测试集
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 从训练集中划分验证集
        valid_size = int(len(X_train) * 0.1)
        X_train_final = X_train.iloc[:-valid_size]
        y_train_final = y_train.iloc[:-valid_size]
        X_valid = X_train.iloc[-valid_size:]
        y_valid = y_train.iloc[-valid_size:]
        
        logger.info(f"数据划分 - 训练: {len(X_train_final)}, 验证: {len(X_valid)}, 测试: {len(X_test)}")
        
        # 训练
        self.train(
            X_train_final, y_train_final,
            X_valid, y_valid,
            log_mlflow=log_mlflow,
            **kwargs
        )
        
        # 在测试集上评估
        test_pred = self.model.predict(X_test)
        test_metrics = calculate_metrics(y_test, test_pred)
        
        logger.info(f"测试集指标: {format_metrics(test_metrics)}")
        
        if log_mlflow:
            with mlflow.start_run(nested=True):
                for metric_name, metric_value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        return self.model, test_metrics
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        self.model.save(filepath)
        logger.info(f"模型已保存到: {filepath}")
