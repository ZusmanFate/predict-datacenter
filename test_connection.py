import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).parent))

from src.data.database import get_db_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_connection():
    """测试数据库连接"""
    try:
        logger.info("=" * 60)
        logger.info("测试 Impala 数据库连接...")
        logger.info("=" * 60)
        
        # 获取数据库管理器
        db_manager = get_db_manager()
        
        # 测试简单查询
        test_query = "SELECT 1 as test"
        result = db_manager.execute_query(test_query)
        
        logger.info(f"✓ 连接成功！查询结果: {result}")
        
        # 测试销量表
        sales_table = db_manager.config['tables']['sales']['name']
        count_query = f"SELECT COUNT(*) as cnt FROM {sales_table} LIMIT 1"
        
        try:
            result = db_manager.execute_query(count_query)
            logger.info(f"✓ 销量表访问成功: {sales_table}")
        except Exception as e:
            logger.warning(f"⚠ 销量表查询失败: {e}")
        
        logger.info("=" * 60)
        logger.info("✅ 数据库连接测试完成！")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 连接失败: {e}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)