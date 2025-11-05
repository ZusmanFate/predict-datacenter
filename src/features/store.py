"""
特征存储管理模块
使用 Feast 管理和服务特征
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """特征存储管理器"""
    
    def __init__(self, repo_path: str = "feature_store"):
        """
        初始化特征存储
        
        Args:
            repo_path: Feast 仓库路径
        """
        self.repo_path = Path(repo_path)
        self.store = None
        
        try:
            from feast import FeatureStore as FeastStore
            self.store = FeastStore(repo_path=str(self.repo_path))
            logger.info(f"Feast 特征存储初始化成功: {repo_path}")
        except ImportError:
            logger.warning("Feast 未安装，特征存储功能不可用。安装: pip install feast")
        except Exception as e:
            logger.warning(f"Feast 特征存储初始化失败: {e}")
    
    def materialize_features(
        self,
        start_date: str,
        end_date: str,
        feature_views: Optional[List[str]] = None
    ) -> None:
        """
        物化特征到在线存储
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            feature_views: 特征视图列表（默认全部）
        """
        if self.store is None:
            logger.error("特征存储未初始化")
            return
        
        try:
            from datetime import datetime
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            logger.info(f"开始物化特征: {start_date} 到 {end_date}")
            
            if feature_views:
                for fv_name in feature_views:
                    self.store.materialize(
                        start_date=start_dt,
                        end_date=end_dt,
                        feature_views=[fv_name]
                    )
                    logger.info(f"✓ 物化完成: {fv_name}")
            else:
                self.store.materialize(start_date=start_dt, end_date=end_dt)
                logger.info("✓ 全部特征物化完成")
            
        except Exception as e:
            logger.error(f"特征物化失败: {e}")
            raise
    
    def get_online_features(
        self,
        entity_rows: List[Dict],
        features: List[str]
    ) -> pd.DataFrame:
        """
        从在线存储获取特征
        
        Args:
            entity_rows: 实体行列表，例如 [{"gcode": "D001", "cust_name": "H001"}]
            features: 特征列表，例如 ["sales_features:sales_quantity_lag_1"]
            
        Returns:
            特征 DataFrame
        """
        if self.store is None:
            logger.error("特征存储未初始化")
            return pd.DataFrame()
        
        try:
            feature_vector = self.store.get_online_features(
                features=features,
                entity_rows=entity_rows
            )
            
            df = feature_vector.to_df()
            logger.info(f"获取在线特征: {len(df)} 行, {len(df.columns)} 列")
            return df
            
        except Exception as e:
            logger.error(f"获取在线特征失败: {e}")
            raise
    
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        从离线存储获取历史特征（用于训练）
        
        Args:
            entity_df: 实体 DataFrame，必须包含实体键和 event_timestamp
            features: 特征列表
            
        Returns:
            历史特征 DataFrame
        """
        if self.store is None:
            logger.error("特征存储未初始化")
            return pd.DataFrame()
        
        try:
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=features
            ).to_df()
            
            logger.info(f"获取历史特征: {len(training_df)} 行, {len(training_df.columns)} 列")
            return training_df
            
        except Exception as e:
            logger.error(f"获取历史特征失败: {e}")
            raise
    
    def save_features_to_offline(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        保存特征到离线存储（Parquet 文件）
        
        Args:
            df: 特征 DataFrame
            output_path: 输出路径
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(output_path, index=False)
            logger.info(f"特征已保存到离线存储: {output_path}")
            
        except Exception as e:
            logger.error(f"保存特征失败: {e}")
            raise
    
    def list_feature_views(self) -> List[str]:
        """
        列出所有特征视图
        
        Returns:
            特征视图名称列表
        """
        if self.store is None:
            logger.error("特征存储未初始化")
            return []
        
        try:
            feature_views = self.store.list_feature_views()
            fv_names = [fv.name for fv in feature_views]
            logger.info(f"特征视图列表: {fv_names}")
            return fv_names
            
        except Exception as e:
            logger.error(f"获取特征视图列表失败: {e}")
            return []


def prepare_features_for_feast(
    df: pd.DataFrame,
    gcode_col: str = 'gcode',
    cust_name_col: str = 'cust_name',
    timestamp_col: str = 'create_dt'
) -> pd.DataFrame:
    """
    准备特征数据用于 Feast
    
    Args:
        df: 原始特征 DataFrame
        gcode_col: 药品编码列名
        cust_name_col: 客户名称列名
        timestamp_col: 时间戳列名
        
    Returns:
        准备好的 DataFrame
    """
    df_feast = df.copy()
    
    # 确保必要的列存在
    required_cols = [gcode_col, cust_name_col, timestamp_col]
    for col in required_cols:
        if col not in df_feast.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    # 重命名为 Feast 标准名称
    df_feast = df_feast.rename(columns={
        gcode_col: 'gcode',
        cust_name_col: 'cust_name',
        timestamp_col: 'event_timestamp'
    })
    
    # 确保时间戳是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df_feast['event_timestamp']):
        df_feast['event_timestamp'] = pd.to_datetime(df_feast['event_timestamp'])
    
    logger.info(f"准备 Feast 特征数据: {len(df_feast)} 行")
    return df_feast
