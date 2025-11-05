"""
调试 SQL 查询语句
查看实际执行的查询并测试不同的过滤条件
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.database import get_db_manager
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def debug_query():
    """调试查询语句"""
    try:
        print("\n" + "=" * 100)
        print(" SQL 查询调试 ".center(100, "="))
        print("=" * 100)
        
        db_manager = get_db_manager()
        table_name = "datasense_dlink_erpservice.view_dws_erp_sal_detail_df"
        
        # ========== 测试1: 不加日期过滤 ==========
        print("\n[测试 1] gcode=026436，无日期过滤")
        print("-" * 100)
        
        query1 = f"""
        SELECT COUNT(*) as count
        FROM {table_name}
        WHERE gcode = '026436'
        """
        
        print("📋 SQL:")
        print(query1)
        result1 = pd.read_sql(query1, db_manager.engine)
        print(f"✅ 结果: {result1['count'].iloc[0]} 条")
        
        # ========== 测试2: 只用 create_dt 过滤 ==========
        print("\n[测试 2] gcode=026436，只用 create_dt 过滤 (>= '2020-01-01' AND <= '2024-12-31')")
        print("-" * 100)
        
        query2 = f"""
        SELECT COUNT(*) as count
        FROM {table_name}
        WHERE gcode = '026436'
          AND create_dt >= '2020-01-01'
          AND create_dt <= '2024-12-31'
        """
        
        print("📋 SQL:")
        print(query2)
        result2 = pd.read_sql(query2, db_manager.engine)
        print(f"✅ 结果: {result2['count'].iloc[0]} 条")
        
        # ========== 测试3: 检查 dt 列 ==========
        print("\n[测试 3] 查看 dt 列的格式")
        print("-" * 100)
        
        query3 = f"""
        SELECT dt, COUNT(*) as count
        FROM {table_name}
        WHERE gcode = '026436'
        GROUP BY dt
        ORDER BY dt DESC
        LIMIT 10
        """
        
        print("📋 SQL:")
        print(query3)
        result3 = pd.read_sql(query3, db_manager.engine)
        print("📊 dt 列样本:")
        print(result3.to_string(index=False))
        
        # ========== 测试4: 用 dt 分区列过滤（YYYYMMDD 格式） ==========
        print("\n[测试 4] gcode=026436，用 dt 分区列过滤 (dt BETWEEN '20200101' AND '20241231')")
        print("-" * 100)
        
        query4 = f"""
        SELECT COUNT(*) as count
        FROM {table_name}
        WHERE gcode = '026436'
          AND dt BETWEEN '20200101' AND '20241231'
        """
        
        print("📋 SQL:")
        print(query4)
        result4 = pd.read_sql(query4, db_manager.engine)
        print(f"✅ 结果: {result4['count'].iloc[0]} 条")
        
        # ========== 测试5: 同时用 dt 和 create_dt 过滤 ==========
        print("\n[测试 5] gcode=026436，同时用 dt 和 create_dt 过滤")
        print("-" * 100)
        
        query5 = f"""
        SELECT COUNT(*) as count
        FROM {table_name}
        WHERE gcode = '026436'
          AND dt BETWEEN '20200101' AND '20241231'
          AND create_dt >= '2020-01-01'
          AND create_dt <= '2024-12-31'
        """
        
        print("📋 SQL:")
        print(query5)
        result5 = pd.read_sql(query5, db_manager.engine)
        print(f"✅ 结果: {result5['count'].iloc[0]} 条")
        
        # ========== 总结 ==========
        print("\n" + "=" * 100)
        print("📊 测试结果总结:")
        print("-" * 100)
        print(f"  测试1 (无日期过滤):                 {result1['count'].iloc[0]:>8,} 条")
        print(f"  测试2 (只用 create_dt):             {result2['count'].iloc[0]:>8,} 条")
        print(f"  测试4 (只用 dt 分区):               {result4['count'].iloc[0]:>8,} 条")
        print(f"  测试5 (dt + create_dt 双重过滤):   {result5['count'].iloc[0]:>8,} 条")
        
        print("\n💡 建议:")
        if result2['count'].iloc[0] > 0:
            print("  ✓ 使用 create_dt 过滤可以正常工作")
            print("  ✓ 建议在查询时不使用 dt 分区列过滤，或确保 dt 列格式正确")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 调试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = debug_query()
    sys.exit(0 if success else 1)
