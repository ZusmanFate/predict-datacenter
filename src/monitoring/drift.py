"""
数据漂移检测模块
监控数据分布变化和模型性能下降
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """数据漂移检测器"""
    
    def __init__(self, threshold: float = 0.05):
        """
        初始化漂移检测器
        
        Args:
            threshold: 显著性水平阈值
        """
        self.threshold = threshold
        self.baseline_stats = {}
        
        logger.info(f"漂移检测器初始化，阈值: {threshold}")
    
    def fit_baseline(self, df: pd.DataFrame, columns: list):
        """
        拟合基线数据统计
        
        Args:
            df: 基线数据
            columns: 要监控的列
        """
        logger.info(f"拟合基线统计: {len(columns)} 个特征")
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"列 {col} 不存在，跳过")
                continue
            
            self.baseline_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        
        logger.info(f"基线统计拟合完成: {len(self.baseline_stats)} 个特征")
    
    def detect_ks_test(
        self,
        baseline: pd.Series,
        current: pd.Series
    ) -> Tuple[bool, float, float]:
        """
        使用 Kolmogorov-Smirnov 检验检测分布漂移
        
        Args:
            baseline: 基线数据
            current: 当前数据
            
        Returns:
            (是否漂移, 统计量, p值)
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        is_drift = p_value < self.threshold
        
        return is_drift, statistic, p_value
    
    def detect_chi2_test(
        self,
        baseline: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> Tuple[bool, float, float]:
        """
        使用卡方检验检测分类变量漂移
        
        Args:
            baseline: 基线数据
            current: 当前数据
            bins: 分箱数（用于连续变量）
            
        Returns:
            (是否漂移, 统计量, p值)
        """
        # 对连续变量分箱
        all_data = pd.concat([baseline, current])
        bins_edges = pd.qcut(all_data, q=bins, duplicates='drop', retbins=True)[1]
        
        baseline_binned = pd.cut(baseline, bins=bins_edges, include_lowest=True)
        current_binned = pd.cut(current, bins=bins_edges, include_lowest=True)
        
        # 构建列联表
        baseline_counts = baseline_binned.value_counts().sort_index()
        current_counts = current_binned.value_counts().sort_index()
        
        # 对齐索引
        all_bins = baseline_counts.index.union(current_counts.index)
        baseline_counts = baseline_counts.reindex(all_bins, fill_value=0)
        current_counts = current_counts.reindex(all_bins, fill_value=0)
        
        # 卡方检验
        contingency_table = pd.DataFrame({
            'baseline': baseline_counts,
            'current': current_counts
        })
        
        statistic, p_value, _, _ = stats.chi2_contingency(contingency_table.T)
        is_drift = p_value < self.threshold
        
        return is_drift, statistic, p_value
    
    def detect_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: Optional[list] = None,
        method: str = 'ks'
    ) -> Dict[str, Dict]:
        """
        检测数据漂移
        
        Args:
            baseline_df: 基线数据
            current_df: 当前数据
            columns: 要检测的列（默认所有数值列）
            method: 检测方法 ('ks', 'chi2')
            
        Returns:
            漂移检测结果
        """
        if columns is None:
            columns = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"开始数据漂移检测: {len(columns)} 个特征，方法: {method}")
        
        results = {}
        drift_count = 0
        
        for col in columns:
            if col not in baseline_df.columns or col not in current_df.columns:
                logger.warning(f"列 {col} 在数据中不存在，跳过")
                continue
            
            try:
                if method == 'ks':
                    is_drift, statistic, p_value = self.detect_ks_test(
                        baseline_df[col],
                        current_df[col]
                    )
                elif method == 'chi2':
                    is_drift, statistic, p_value = self.detect_chi2_test(
                        baseline_df[col],
                        current_df[col]
                    )
                else:
                    raise ValueError(f"不支持的方法: {method}")
                
                results[col] = {
                    'is_drift': is_drift,
                    'statistic': statistic,
                    'p_value': p_value,
                    'method': method
                }
                
                if is_drift:
                    drift_count += 1
                    logger.warning(f"检测到漂移: {col} (p={p_value:.6f})")
            
            except Exception as e:
                logger.error(f"检测 {col} 时出错: {e}")
                results[col] = {
                    'is_drift': None,
                    'error': str(e)
                }
        
        logger.info(f"漂移检测完成: {drift_count}/{len(columns)} 个特征发生漂移")
        
        return results
    
    def detect_performance_drift(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        tolerance: float = 0.1
    ) -> Dict[str, bool]:
        """
        检测模型性能漂移
        
        Args:
            baseline_metrics: 基线性能指标
            current_metrics: 当前性能指标
            tolerance: 容忍度（相对变化）
            
        Returns:
            性能漂移结果
        """
        logger.info("检测模型性能漂移...")
        
        drift_results = {}
        
        for metric_name in baseline_metrics.keys():
            if metric_name not in current_metrics:
                logger.warning(f"指标 {metric_name} 在当前数据中不存在")
                continue
            
            baseline_value = baseline_metrics[metric_name]
            current_value = current_metrics[metric_name]
            
            # 计算相对变化
            if baseline_value != 0:
                relative_change = abs((current_value - baseline_value) / baseline_value)
            else:
                relative_change = abs(current_value - baseline_value)
            
            is_drift = relative_change > tolerance
            
            drift_results[metric_name] = {
                'is_drift': is_drift,
                'baseline': baseline_value,
                'current': current_value,
                'relative_change': relative_change,
                'absolute_change': current_value - baseline_value
            }
            
            if is_drift:
                logger.warning(
                    f"性能漂移: {metric_name} "
                    f"({baseline_value:.4f} -> {current_value:.4f}, "
                    f"变化: {relative_change*100:.2f}%)"
                )
        
        return drift_results
    
    def generate_drift_report(
        self,
        drift_results: Dict,
        output_path: str = "reports/drift_report.txt"
    ):
        """
        生成漂移检测报告
        
        Args:
            drift_results: 漂移检测结果
            output_path: 报告输出路径
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("数据漂移检测报告\n")
            f.write("=" * 80 + "\n\n")
            
            drift_features = [k for k, v in drift_results.items() if v.get('is_drift')]
            f.write(f"检测特征总数: {len(drift_results)}\n")
            f.write(f"发生漂移特征: {len(drift_features)}\n")
            f.write(f"漂移比例: {len(drift_features)/len(drift_results)*100:.2f}%\n\n")
            
            if drift_features:
                f.write("漂移特征详情:\n")
                f.write("-" * 80 + "\n")
                for feature in drift_features:
                    result = drift_results[feature]
                    f.write(f"\n特征: {feature}\n")
                    f.write(f"  方法: {result.get('method', 'N/A')}\n")
                    f.write(f"  统计量: {result.get('statistic', 'N/A'):.6f}\n")
                    f.write(f"  P值: {result.get('p_value', 'N/A'):.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"报告生成时间: {pd.Timestamp.now()}\n")
        
        logger.info(f"漂移报告已生成: {output_path}")
