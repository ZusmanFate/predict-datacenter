"""
测试 StarRocks 数据库连接和表结构
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.database import get_db_manager
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


def test_starrocks_connection():
    """测试 StarRocks 连接"""
    try:
        logger.info("=" * 80)
        logger.info("测试 StarRocks 数据库连接")
        logger.info("=" * 80)
        
        # 获取数据库管理器
        db_manager = get_db_manager()
        
        logger.info(f"数据库类型: {db_manager.db_type}")
        logger.info(f"数据库配置: {db_manager.config.get('starrocks', {}).get('database')}")
        
        # 1. 测试基本连接
        logger.info("\n[测试 1/4] 基本连接测试...")
        test_query = "SELECT 1 as test"
        result = db_manager.execute_query(test_query)
        logger.info(f"✓ 连接成功！查询结果: {result}")
        
        # 2. 查看当前数据库
        logger.info("\n[测试 2/4] 查看当前数据库...")
        db_query = "SELECT DATABASE() as current_db"
        result = db_manager.execute_query(db_query)
        logger.info(f"当前数据库: {result}")
        
        # 3. 测试销量表是否存在
        logger.info("\n[测试 3/4] 测试销量表访问...")
        sales_table = db_manager.config['tables']['sales']['name']
        logger.info(f"销量表名: {sales_table}")
        
        # 查询表结构
        desc_query = f"DESCRIBE {sales_table}"
        try:
            columns_df = pd.read_sql(desc_query, db_manager.engine)
            logger.info(f"✓ 表结构查询成功！")
            logger.info(f"\n表字段列表（共 {len(columns_df)} 个字段）:")
            print(columns_df.to_string(index=False))
        except Exception as e:
            logger.error(f"✗ 表结构查询失败: {e}")
            return False
        
        # 4. 查询少量数据测试
        logger.info("\n[测试 4/4] 查询数据测试...")
        sample_query = f"SELECT * FROM {sales_table} LIMIT 5"
        try:
            sample_df = pd.read_sql(sample_query, db_manager.engine)
            logger.info(f"✓ 数据查询成功！返回 {len(sample_df)} 条记录")
            logger.info(f"\n数据列名:")
            print(sample_df.columns.tolist())
            logger.info(f"\n数据样本:")
            print(sample_df.head())
            
            # 统计信息
            logger.info(f"\n数据统计:")
            count_query = f"SELECT COUNT(*) as total_count FROM {sales_table}"
            count_result = db_manager.execute_query(count_query)
            logger.info(f"  总记录数: {count_result}")
            
        except Exception as e:
            logger.error(f"✗ 数据查询失败: {e}")
            return False
        
        # 5. 检查关键字段
        logger.info("\n[额外检查] 检查关键字段...")
        expected_columns = ['gcode', 'create_dt', 'qty', 'cust_name']
        actual_columns = sample_df.columns.tolist()
        
        for col in expected_columns:
            if col in actual_columns:
                logger.info(f"  ✓ 字段 '{col}' 存在")
            else:
                logger.warning(f"  ⚠ 字段 '{col}' 不存在")
                # 尝试模糊匹配
                similar = [c for c in actual_columns if col.lower() in c.lower()]
                if similar:
                    logger.info(f"    可能的替代字段: {similar}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ StarRocks 连接测试完成！")
        logger.info("=" * 80)
        
        logger.info("\n建议下一步:")
        logger.info("  1. 查看上面的字段列表")
        logger.info("  2. 更新 config/database.yaml 中的 columns 配置")
        logger.info("  3. 更新 src/data/loader.py 中的字段映射")
        logger.info("  4. 运行 quick_test.py 验证完整流程")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_starrocks_connection()
    sys.exit(0 if success else 1)
