"""
检查数据库表结构
查看实际的列名和数据类型
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.database import get_db_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def check_table_structure():
    """检查表结构"""
    try:
        logger.info("=" * 60)
        logger.info("检查数据库表结构...")
        logger.info("=" * 60)
        
        db_manager = get_db_manager()
        
        # 检查销售数据表结构
        sales_table = "datasense_dlink_erpservice.view_dws_erp_sal_detail_df"
        
        logger.info(f"\n查询表结构: {sales_table}")
        
        # 使用 DESCRIBE 或 SHOW COLUMNS 查看表结构
        try:
            # 方法1: 使用 DESCRIBE
            query = f"DESCRIBE {sales_table}"
            result = db_manager.execute_query(query)
            logger.info(f"\n表结构 (DESCRIBE):")
            for row in result:
                logger.info(f"  {row}")
        except Exception as e1:
            logger.warning(f"DESCRIBE 失败: {e1}")
            
            try:
                # 方法2: 使用 SHOW COLUMNS
                query = f"SHOW COLUMNS FROM {sales_table}"
                result = db_manager.execute_query(query)
                logger.info(f"\n表结构 (SHOW COLUMNS):")
                for row in result:
                    logger.info(f"  {row}")
            except Exception as e2:
                logger.warning(f"SHOW COLUMNS 失败: {e2}")
                
                # 方法3: 查询少量数据并获取列名
                logger.info("尝试通过查询获取列名...")
                query = f"SELECT * FROM {sales_table} LIMIT 1"
                import pandas as pd
                df = pd.read_sql(query, db_manager.engine)
                logger.info(f"\n实际列名 ({len(df.columns)} 列):")
                for i, col in enumerate(df.columns, 1):
                    logger.info(f"  {i}. {col}")
                
                # 保存列名到文件
                with open("actual_columns.txt", "w", encoding="utf-8") as f:
                    f.write("# 实际数据库视图列名\n")
                    f.write("# 可以复制到 database.yaml 的 sales.columns 中\n\n")
                    for col in df.columns:
                        f.write(f"      - {col}\n")
                
                logger.info(f"\n✓ 列名已保存到 actual_columns.txt")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 表结构检查完成！")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 检查失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = check_table_structure()
    sys.exit(0 if success else 1)
