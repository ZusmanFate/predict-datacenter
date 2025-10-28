"""
数据加载模块
从数据库或文件加载销量数据
"""
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.database import get_db_manager
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config_path: str = "config/database.yaml"):
        """
        初始化数据加载器
        
        Args:
            config_path: 数据库配置文件路径
        """
        self.db_manager = get_db_manager(config_path)
        self.config = load_config(config_path)
        self.table_config = self.config['tables']
    
    def load_sales_data(
        self,
        drug_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        从数据库加载销量数据
        
        Args:
            drug_id: 药品ID（可选）
            hospital_id: 医院ID（可选）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            limit: 限制返回行数
            
        Returns:
            销量数据 DataFrame
        """
        table_name = self.table_config['sales']['name']
        
        # 构建查询
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        
        if drug_id:
            query += f" AND drug_id = '{drug_id}'"
        
        if hospital_id:
            query += f" AND hospital_id = '{hospital_id}'"
        
        if start_date:
            query += f" AND date >= '{start_date}'"
        
        if end_date:
            query += f" AND date <= '{end_date}'"
        
        query += " ORDER BY date ASC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"加载销量数据: drug_id={drug_id}, hospital_id={hospital_id}, date_range=[{start_date}, {end_date}]")
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"成功加载 {len(df)} 条销量数据")
            return df
        except Exception as e:
            logger.error(f"加载销量数据失败: {e}")
            raise
    
    def load_drug_info(self, drug_id: Optional[str] = None) -> pd.DataFrame:
        """
        加载药品信息
        
        Args:
            drug_id: 药品ID（可选）
            
        Returns:
            药品信息 DataFrame
        """
        table_name = self.table_config['drugs']['name']
        query = f"SELECT * FROM {table_name}"
        
        if drug_id:
            query += f" WHERE drug_id = '{drug_id}'"
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            logger.info(f"成功加载 {len(df)} 条药品信息")
            return df
        except Exception as e:
            logger.error(f"加载药品信息失败: {e}")
            raise
    
    def load_hospital_info(self, hospital_id: Optional[str] = None) -> pd.DataFrame:
        """
        加载医院信息
        
        Args:
            hospital_id: 医院ID（可选）
            
        Returns:
            医院信息 DataFrame
        """
        table_name = self.table_config['hospitals']['name']
        query = f"SELECT * FROM {table_name}"
        
        if hospital_id:
            query += f" WHERE hospital_id = '{hospital_id}'"
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            logger.info(f"成功加载 {len(df)} 条医院信息")
            return df
        except Exception as e:
            logger.error(f"加载医院信息失败: {e}")
            raise
    
    def load_batch_data(
        self,
        drug_ids: Optional[List[str]] = None,
        hospital_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        批量加载多个药品和医院的数据
        
        Args:
            drug_ids: 药品ID列表
            hospital_ids: 医院ID列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            批量销量数据 DataFrame
        """
        table_name = self.table_config['sales']['name']
        
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        
        if drug_ids:
            drug_ids_str = "', '".join(drug_ids)
            query += f" AND drug_id IN ('{drug_ids_str}')"
        
        if hospital_ids:
            hospital_ids_str = "', '".join(hospital_ids)
            query += f" AND hospital_id IN ('{hospital_ids_str}')"
        
        if start_date:
            query += f" AND date >= '{start_date}'"
        
        if end_date:
            query += f" AND date <= '{end_date}'"
        
        query += " ORDER BY drug_id, hospital_id, date ASC"
        
        logger.info(f"批量加载数据: {len(drug_ids or [])} 个药品, {len(hospital_ids or [])} 个医院")
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"成功加载 {len(df)} 条批量数据")
            return df
        except Exception as e:
            logger.error(f"批量加载数据失败: {e}")
            raise
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        从 CSV 文件加载数据
        
        Args:
            filepath: CSV 文件路径
            
        Returns:
            数据 DataFrame
        """
        logger.info(f"从 CSV 加载数据: {filepath}")
        try:
            df = pd.read_csv(filepath)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            logger.info(f"成功从 CSV 加载 {len(df)} 条数据")
            return df
        except Exception as e:
            logger.error(f"从 CSV 加载数据失败: {e}")
            raise
    
    def load_from_excel(self, filepath: str, sheet_name: str = 0) -> pd.DataFrame:
        """
        从 Excel 文件加载数据
        
        Args:
            filepath: Excel 文件路径
            sheet_name: 工作表名称或索引
            
        Returns:
            数据 DataFrame
        """
        logger.info(f"从 Excel 加载数据: {filepath}, sheet={sheet_name}")
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            logger.info(f"成功从 Excel 加载 {len(df)} 条数据")
            return df
        except Exception as e:
            logger.error(f"从 Excel 加载数据失败: {e}")
            raise
    
    def get_unique_drugs(self) -> List[str]:
        """
        获取所有唯一的药品ID
        
        Returns:
            药品ID列表
        """
        table_name = self.table_config['sales']['name']
        query = f"SELECT DISTINCT drug_id FROM {table_name} ORDER BY drug_id"
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            drug_ids = df['drug_id'].tolist()
            logger.info(f"获取到 {len(drug_ids)} 个唯一药品")
            return drug_ids
        except Exception as e:
            logger.error(f"获取药品列表失败: {e}")
            raise
    
    def get_unique_hospitals(self) -> List[str]:
        """
        获取所有唯一的医院ID
        
        Returns:
            医院ID列表
        """
        table_name = self.table_config['sales']['name']
        query = f"SELECT DISTINCT hospital_id FROM {table_name} ORDER BY hospital_id"
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            hospital_ids = df['hospital_id'].tolist()
            logger.info(f"获取到 {len(hospital_ids)} 个唯一医院")
            return hospital_ids
        except Exception as e:
            logger.error(f"获取医院列表失败: {e}")
            raise
    
    def get_date_range(self, drug_id: str = None, hospital_id: str = None) -> tuple:
        """
        获取数据的日期范围
        
        Args:
            drug_id: 药品ID（可选）
            hospital_id: 医院ID（可选）
            
        Returns:
            (min_date, max_date)
        """
        table_name = self.table_config['sales']['name']
        query = f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table_name} WHERE 1=1"
        
        if drug_id:
            query += f" AND drug_id = '{drug_id}'"
        
        if hospital_id:
            query += f" AND hospital_id = '{hospital_id}'"
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            min_date = df['min_date'].iloc[0]
            max_date = df['max_date'].iloc[0]
            logger.info(f"数据日期范围: {min_date} 到 {max_date}")
            return min_date, max_date
        except Exception as e:
            logger.error(f"获取日期范围失败: {e}")
            raise
