"""
数据加载模块
从数据库或文件加载销量数据
"""
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional, Tuple
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
    
    def _format_date_for_db(self, date_str: str) -> str:
        """
        将日期字符串格式化为数据库的 dt 分区列格式 (YYYYMMDD)
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            格式化后的日期字符串 (YYYYMMDD)
        """
        try:
            date_obj = pd.to_datetime(date_str)
            return date_obj.strftime('%Y%m%d')
        except Exception as e:
            logger.warning(f"日期格式化失败: {date_str}, 错误: {e}")
            return date_str.replace('-', '')
    
    def load_data(self, table_name: str, **kwargs) -> pd.DataFrame:
        """
        从数据库加载数据
        
        Args:
            table_name: 表名
            **kwargs: 查询参数
            
        Returns:
            DataFrame
        """
        return self.db_manager.load_data(table_name, **kwargs)

    def load_sales_data(
        self,
        gcode: Optional[str] = None,  # 对应原 drug_id
        cust_name: Optional[str] = None,  # 对应原 hospital_id
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        dt_filter_date: Optional[str] = None,  # dt 分区列过滤 (YYYY-MM-DD 格式，DATE类型)
        use_yesterday_dt: bool = False,  # 是否使用昨天作为 dt 过滤
        use_last_5years: bool = False,  # 是否默认使用近5年数据
        limit: Optional[int] = None,
        # 保持向后兼容
        drug_id: Optional[str] = None,
        hospital_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从数据库加载销量数据
        
        Args:
            gcode: 药品编码
            cust_name: 客户名称
            start_date: 开始日期 (YYYY-MM-DD)，用于筛选 create_dt（开单日期）
            end_date: 结束日期 (YYYY-MM-DD)，用于筛选 create_dt（开单日期）
            dt_filter_date: dt 分区字段过滤 (YYYY-MM-DD 格式)
                          dt 是数据采集日期（按天全量采集），不是开单日期
                          例如：dt='2025-11-04' 表示从昨天的全量分区读数据
            use_yesterday_dt: 是否使用昨天作为 dt 分区过滤（默认 False）
            use_last_5years: 是否默认使用近5年数据（默认 False）
                           仅影响 create_dt 的过滤范围
            limit: 限制返回行数
            
        Returns:
            销量数据 DataFrame
            
        重要说明：
            - **dt**: 分区字段，表示数据采集日期（按天全量），格式 YYYY-MM-DD
                     例如 dt='2025-11-04' 表示读取 2025-11-04 这天采集的全量数据
            - **create_dt**: 开单日期（业务日期），实际销售发生的日期
                           start_date/end_date 用于筛选 create_dt
            - 查询逻辑：WHERE dt='2025-11-04' AND create_dt BETWEEN '2020-01-01' AND '2024-12-31'
        """
        from datetime import datetime, timedelta
        
        # 向后兼容性：将旧参数映射到新参数
        if drug_id is not None and gcode is None:
            gcode = drug_id
            logger.info(f"使用向后兼容模式：drug_id={drug_id} -> gcode={gcode}")
        if hospital_id is not None and cust_name is None:
            cust_name = hospital_id
            logger.info(f"使用向后兼容模式：hospital_id={hospital_id} -> cust_name={cust_name}")
        
        # 处理 dt 默认值：使用昨天（分区过滤）
        if use_yesterday_dt and dt_filter_date is None:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            dt_filter_date = yesterday
            logger.info(f"使用昨天作为 dt 分区过滤: {dt_filter_date}")
        
        # 处理默认日期范围：近5年（仅影响 create_dt 过滤）
        if use_last_5years:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
                logger.info(f"使用近5年数据，create_dt start_date 设置为: {start_date}")
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                logger.info(f"create_dt end_date 设置为今天: {end_date}")
        
        table_name = self.table_config['sales']['name']

        columns_to_select = ", ".join(self.table_config['sales']['columns'])
        # 构建查询
        query = f"SELECT {columns_to_select} FROM {table_name} WHERE 1=1"
        
        # 添加 dt 分区字段过滤（只用于指定从哪天的全量分区读数据）
        if dt_filter_date:
            # dt 是分区字段，格式 YYYY-MM-DD，表示数据采集日期
            query += f" AND dt = '{dt_filter_date}'"
            logger.info(f"dt 分区过滤: dt = '{dt_filter_date}' (从这天的全量分区读数据)")  
        
        if gcode:
            query += f" AND gcode = '{gcode}'"
        
        if cust_name:
            query += f" AND cust_name = '{cust_name}'"
        
        # create_dt 字段过滤（开单日期，业务日期）
        if start_date:
            query += f" AND create_dt >= '{start_date}'"
            logger.info(f"create_dt 过滤: create_dt >= '{start_date}' (开单日期)")
        
        if end_date:
            query += f" AND create_dt <= '{end_date}'"
            logger.info(f"create_dt 过滤: create_dt <= '{end_date}' (开单日期)")
        
        query += " ORDER BY create_dt ASC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"加载销量数据: gcode={gcode}, cust_name={cust_name}, date_range=[{start_date}, {end_date}], dt_filter={dt_filter_date}")
        logger.debug(f"SQL查询: {query}")
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            # 将 create_dt 转换为 datetime 对象
            if 'create_dt' in df.columns:
                df['create_dt'] = pd.to_datetime(df['create_dt'])
            
            logger.info(f"成功加载 {len(df)} 条销量数据")
            return df
        except Exception as e:
            logger.error(f"加载销量数据失败: {e}")
            raise
    
    def load_drug_info(self, drug_id: Optional[str] = None) -> pd.DataFrame:
        """
        加载药品信息 (dim.dim_erp_mst_goods_info_df)
        
        Args:
            gcode: 药品编码（对应 drug_id）
            
        Returns:
            药品信息 DataFrame
        """
        table_name = self.table_config['drugs']['name']
        columns_to_select = ", ".join(self.table_config['drugs']['columns'])
        query = f"SELECT {columns_to_select} FROM {table_name} WHERE 1=1"
        
        if gcode:
            query += f" AND gcode = '{gcode}'"
        
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
        gcodes: Optional[List[str]] = None, # 对应 drug_ids
        cust_names: Optional[List[str]] = None, # 对应 hospital_ids
        start_date: Optional[str] = None, # YYYY-MM-DD
        end_date: Optional[str] = None,   # YYYY-MM-DD
        dt_filter_date: Optional[str] = None # YYYYMMDD 格式，用于 dt 分区列过滤
    ) -> pd.DataFrame:
        """
        批量加载多个药品和医院的销量数据
        
        Args:
            gcodes: 药品编码列表
            cust_names: 客户名称列表
            start_date: 销售发生开始日期 (YYYY-MM-DD)
            end_date: 销售发生结束日期 (YYYY-MM-DD)
            dt_filter_date: 用于 dt 分区列的日期过滤 (YYYYMMDD)。
            
        Returns:
            批量销量数据 DataFrame
        """
        table_name = self.table_config['sales']['name']
        columns_to_select = ", ".join(self.table_config['sales']['columns'])

        query = f"SELECT {columns_to_select} FROM {table_name} WHERE 1=1"
        
        if dt_filter_date:
            query += f" AND dt = '{dt_filter_date}'"
        elif start_date and end_date and self.db_manager.db_type in ['impala', 'starrocks']:
            formatted_start_dt = self._format_date_for_db(start_date)
            formatted_end_dt = self._format_date_for_db(end_date)
            query += f" AND dt BETWEEN '{formatted_start_dt}' AND '{formatted_end_dt}'"
        elif start_date and self.db_manager.db_type in ['impala', 'starrocks']:
             formatted_start_dt = self._format_date_for_db(start_date)
             query += f" AND dt >= '{formatted_start_dt}'"
        elif end_date and self.db_manager.db_type in ['impala', 'starrocks']:
             formatted_end_dt = self._format_date_for_db(end_date)
             query += f" AND dt <= '{formatted_end_dt}'"

        if gcodes:
            gcodes_str = "', '".join(gcodes)
            query += f" AND gcode IN ('{gcodes_str}')"
        
        if cust_names:
            cust_names_str = "', '".join(cust_names)
            query += f" AND cust_name IN ('{cust_names_str}')"
        
        if start_date:
            query += f" AND create_dt >= '{start_date}'"
        
        if end_date:
            query += f" AND create_dt <= '{end_date}'"
        
        query += " ORDER BY gcode, cust_name, create_dt ASC"
        
        logger.info(f"批量加载数据: {len(gcodes or [])} 个药品, {len(cust_names or [])} 个医院")
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            df['create_dt'] = pd.to_datetime(df['create_dt'])
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
            # 尝试将常见的日期列转换为 datetime
            for col in ['date', 'create_dt', 'prod_dt', 'valid_dt']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except ValueError:
                        logger.warning(f"列 '{col}' 无法完全转换为日期格式，跳过。")
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
            # 尝试将常见的日期列转换为 datetime
            for col in ['date', 'create_dt', 'prod_dt', 'valid_dt']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except ValueError:
                        logger.warning(f"列 '{col}' 无法完全转换为日期格式，跳过。")
            logger.info(f"成功从 Excel 加载 {len(df)} 条数据")
            return df
        except Exception as e:
            logger.error(f"从 Excel 加载数据失败: {e}")
            raise
    
    def get_unique_gcodes(self) -> List[str]:
        """
        获取所有唯一的药品编码
        
        Returns:
            药品编码列表
        """
        table_name = self.table_config['sales']['name']

        query = f"SELECT DISTINCT gcode FROM {table_name} ORDER BY gcode"
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            gcodes = df['gcode'].tolist()
            logger.info(f"获取到 {len(gcodes)} 个唯一药品编码")
            return gcodes
        except Exception as e:
            logger.error(f"获取药品编码列表失败: {e}")
            raise
    
    def get_unique_hospitals(self) -> List[str]:
        """
        获取所有唯一的客户名称 (cust_name)
        
        Returns:
            客户名称列表
        """
        table_name = self.table_config['sales']['name']
        query = f"SELECT DISTINCT cust_name FROM {table_name} ORDER BY cust_name"
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            cust_names = df['cust_name'].tolist()
            logger.info(f"获取到 {len(cust_names)} 个唯一客户名称")
            return cust_names
        except Exception as e:
            logger.error(f"获取客户名称列表失败: {e}")
            raise
    
    def get_sales_date_range(self, gcode: Optional[str] = None, cust_name: Optional[str] = None) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        获取销量数据的日期范围 (基于 create_dt)
        
        Args:
            gcode: 药品编码（可选）
            cust_name: 客户名称（可选）
            
        Returns:
            (min_create_dt, max_create_dt)
        """
        table_name = self.table_config['sales']['name']
        query = f"SELECT MIN(create_dt) as min_dt, MAX(create_dt) as max_dt FROM {table_name} WHERE 1=1"
        
        # 对于分区表，最好也加上 dt 过滤，否则可能扫描全表
        # 这里为了简化，不加 dt 过滤，但实际生产中可能需要
        
        if gcode:
            query += f" AND gcode = '{gcode}'"
        
        if cust_name:
            query += f" AND cust_name = '{cust_name}'"
        
        try:
            df = pd.read_sql(query, self.db_manager.engine)
            min_dt = pd.to_datetime(df['min_dt'].iloc[0]) if not pd.isna(df['min_dt'].iloc[0]) else None
            max_dt = pd.to_datetime(df['max_dt'].iloc[0]) if not pd.isna(df['max_dt'].iloc[0]) else None
            logger.info(f"销量数据日期范围: {min_dt} 到 {max_dt}")
            return min_dt, max_dt
        except Exception as e:
            logger.error(f"获取销量日期范围失败: {e}")
            raise