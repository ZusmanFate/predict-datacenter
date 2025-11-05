"""
数据库连接模块
支持 MySQL, PostgreSQL, SQLite
"""
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Optional, Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config_path: str = "config/database.yaml"):
        """
        初始化数据库管理器
        
        Args:
            config_path: 数据库配置文件路径
        """
        self.config = load_config(config_path)
        self.db_type = self.config.get('default', 'sqlite')
        self.engine = None
        self.SessionLocal = None
        self._connect()
    
    def _connect(self) -> None:
        """建立数据库连接"""
        try:
            connection_string = self._get_connection_string()
            logger.info(f"正在连接到 {self.db_type} 数据库...")
            
            # 创建引擎
            if self.db_type == 'sqlite' or self.db_type == 'impala':
                self.engine = create_engine(
                    connection_string,
                    echo=self.config[self.db_type].get('echo', False)
                )
            else:
                self.engine = create_engine(
                    connection_string,
                    poolclass=QueuePool,
                    pool_size=self.config[self.db_type].get('pool_size', 10),
                    max_overflow=self.config[self.db_type].get('max_overflow', 20),
                    pool_recycle=self.config[self.db_type].get('pool_recycle', 3600),
                    echo=self.config[self.db_type].get('echo', False)
                )
            
            # 创建会话工厂
            self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
            
            logger.info(f"数据库连接成功: {self.db_type}")
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def _get_connection_string(self) -> str:
        """
        获取数据库连接字符串
        
        Returns:
            连接字符串
        """
        db_config = self.config[self.db_type]
        
        if self.db_type == 'sqlite':
            db_path = Path(db_config['database'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_config['database']}"
        
        elif self.db_type == 'mysql':
            return (
                f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
                f"?charset={db_config.get('charset', 'utf8mb4')}"
            )
        
        elif self.db_type == 'postgresql':
            return (
                f"postgresql+psycopg2://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
        elif self.db_type == 'impala':
            host = db_config['host']
            port = db_config['port']
            database = db_config['database']
            username = db_config.get('username', '')
            password = db_config.get('password', '')
            auth_mechanism = db_config.get('auth_mechanism', 'NOSASL')

            # 使用 PyHive 连接 Impala
            # 方案 1: HTTP 协议（推荐，不需要 SASL）
            if auth_mechanism == 'NOSASL' or not username:
                # 使用 HTTP 传输
                conn_str = f"hive+http://{host}:{port}/{database}"
            else:
                # 方案 2: Thrift 协议（需要 SASL）
                conn_str = f"hive+thrift://"
                if username:
                    conn_str += f"{username}"
                    if password:
                        conn_str += f":{password}"
                    conn_str += "@"
                conn_str += f"{host}:{port}/{database}"
            
            logger.info(f"Impala 连接字符串: {conn_str.replace(password, '***') if password else conn_str}")
            return conn_str

        elif self.db_type == 'starrocks':
            # StarRocks 兼容 MySQL 协议，使用 mysql+pymysql 方言
            from urllib.parse import quote_plus
            
            username = quote_plus(db_config['username'])
            password = quote_plus(db_config['password'])
            host = db_config['host']
            port = db_config['port']
            database = db_config['database']
            charset = db_config.get('charset', 'utf8mb4')
            
            conn_str = (
                f"mysql+pymysql://{username}:{password}"
                f"@{host}:{port}/{database}"
                f"?charset={charset}"
            )
            
            logger.info(f"StarRocks 连接字符串: mysql+pymysql://{username}:***@{host}:{port}/{database}")
            return conn_str

        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        获取数据库会话（上下文管理器）
        
        Yields:
            数据库会话
        """
        if self.db_type == 'impala':
            logger.warning("对于 Impala 数据库，通常更推荐使用 execute_query 方法直接执行 SQL，而不是通过 ORM 会话。")
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None):
        """
        执行 SQL 查询
        
        Args:
            query: SQL 查询语句
            params: 查询参数
            
        Returns:
            查询结果
        """
        from sqlalchemy import text
        
        with self.engine.connect() as conn:
            # SQLAlchemy 2.0 需要使用 text() 包装 SQL 语句
            result = conn.execute(text(query), params or {})
            return result.fetchall()
    
    def create_tables(self) -> None:
        """创建数据表"""
        pass
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("数据库连接已关闭")


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(config_path: str = "config/database.yaml") -> DatabaseManager:
    """
    获取数据库管理器单例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        数据库管理器实例
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(config_path)
    return _db_manager
