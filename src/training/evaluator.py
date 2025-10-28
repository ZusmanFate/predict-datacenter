"""
模型评估器
提供多种评估指标和可视化功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.helpers import calculate_metrics, format_metrics

logger = get_logger(__name__)

# 设置中文字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_details: bool = False
    ) -> Dict[str, float]:
        """
        评估预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            return_details: 是否返回详细信息
            
        Returns:
            评估指标字典
        """
        metrics = calculate_metrics(y_true, y_pred)
        
        if return_details:
            # 添加额外的详细指标
            errors = y_true - y_pred
            metrics['mean_error'] = np.mean(errors)
            metrics['std_error'] = np.std(errors)
            metrics['max_error'] = np.max(np.abs(errors))
            metrics['min_error'] = np.min(np.abs(errors))
        
        logger.info(f"评估完成: {format_metrics(metrics)}")
        return metrics
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.Series] = None,
        title: str = "预测 vs 实际",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        绘制预测结果对比图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            dates: 日期序列
            title: 图表标题
            save_path: 保存路径
            figsize: 图形大小
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        x = dates if dates is not None else np.arange(len(y_true))
        
        # 子图1: 实际值 vs 预测值
        axes[0].plot(x, y_true, label='实际值', color='blue', linewidth=2, alpha=0.7)
        axes[0].plot(x, y_pred, label='预测值', color='red', linewidth=2, alpha=0.7)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].set_ylabel('销量', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 子图2: 预测误差
        errors = y_true - y_pred
        axes[1].plot(x, errors, label='预测误差', color='green', linewidth=2, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].fill_between(x, 0, errors, alpha=0.3, color='green')
        axes[1].set_title('预测误差', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('日期' if dates is not None else '样本', fontsize=12)
        axes[1].set_ylabel('误差', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "实际值 vs 预测值散点图",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ):
        """
        绘制散点图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_path: 保存路径
            figsize: 图形大小
        """
        plt.figure(figsize=figsize)
        
        # 散点图
        plt.scatter(y_true, y_pred, alpha=0.5, s=50)
        
        # 对角线（完美预测线）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
        
        plt.xlabel('实际值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 添加R²值
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=plt.gca().transAxes, 
                fontsize=12, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"散点图已保存到: {save_path}")
        
        plt.show()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5)
    ):
        """
        绘制残差分析图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            save_path: 保存路径
            figsize: 图形大小
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 残差分布直方图
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('残差', fontsize=12)
        axes[0].set_ylabel('频数', fontsize=12)
        axes[0].set_title('残差分布', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 残差 vs 预测值
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('预测值', fontsize=12)
        axes[1].set_ylabel('残差', fontsize=12)
        axes[1].set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q图（正态性检验）
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q图', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"残差图已保存到: {save_path}")
        
        plt.show()
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        比较多个模型的性能
        
        Args:
            results: 模型结果字典 {模型名: {指标名: 值}}
            save_path: 保存路径
            figsize: 图形大小
        """
        # 转换为 DataFrame
        df = pd.DataFrame(results).T
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 柱状图
        df.plot(kind='bar', ax=axes[0], rot=45)
        axes[0].set_title('模型性能比较', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('模型', fontsize=12)
        axes[0].set_ylabel('指标值', fontsize=12)
        axes[0].legend(title='指标')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 热力图
        sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[1])
        axes[1].set_title('模型指标热力图', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('指标', fontsize=12)
        axes[1].set_ylabel('模型', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"模型比较图已保存到: {save_path}")
        
        plt.show()
        
        logger.info("\n模型性能比较:")
        print(df.to_string())
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_dir: str = "reports"
    ) -> str:
        """
        生成完整的评估报告
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 计算指标
        metrics = self.evaluate(y_true, y_pred, return_details=True)
        
        # 生成所有图表
        self.plot_predictions(
            y_true, y_pred,
            title=f"{model_name} - 预测 vs 实际",
            save_path=output_path / f"{model_name}_predictions.png"
        )
        
        self.plot_scatter(
            y_true, y_pred,
            title=f"{model_name} - 散点图",
            save_path=output_path / f"{model_name}_scatter.png"
        )
        
        self.plot_residuals(
            y_true, y_pred,
            save_path=output_path / f"{model_name}_residuals.png"
        )
        
        # 生成文本报告
        report_path = output_path / f"{model_name}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"模型评估报告: {model_name}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("评估指标:\n")
            f.write("-" * 60 + "\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"  {metric_name.upper()}: {metric_value:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write(f"报告生成时间: {pd.Timestamp.now()}\n")
            f.write(f"样本数量: {len(y_true)}\n")
        
        logger.info(f"评估报告已生成: {report_path}")
        return str(report_path)
