"""
检查 SQL 查询语句和数据质量
查看 gcode=026436 的数据情况
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.database import get_db_manager
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def check_query_and_data():
    """检查查询语句和数据"""
    try:
        logger.info("=" * 80)
        logger.info("检查 SQL 查询语句和数据质量")
        logger.info("=" * 80)
        
        db_manager = get_db_manager()
        table_name = "datasense_dlink_erpservice.view_dws_erp_sal_detail_df"
        
        # ========== 1. 查看 gcode=026436 的基本信息 ==========
        print("\n[1] 查询 gcode=026436 的基本数据统计")
        print("-" * 80)
        
        query1 = f"""
        SELECT 
            COUNT(*) as total_count,
            MIN(create_dt) as min_date,
            MAX(create_dt) as max_date,
            COUNT(DISTINCT cust_name) as unique_customers,
            SUM(qty) as total_qty,
            AVG(qty) as avg_qty
        FROM {table_name}
        WHERE gcode = '026436'
        """
        
        print("\n📋 执行的 SQL 查询:")
        print(query1)
        
        result = pd.read_sql(query1, db_manager.engine)
        print("\n📊 查询结果:")
        print(result.to_string(index=False))
        
        # ========== 2. 查看最近的数据样本 ==========
        print("\n\n[2] 查询 gcode=026436 最近的数据样本 (最新10条)")
        print("-" * 80)
        
        query2 = f"""
        SELECT 
            gcode,
            create_dt,
            qty,
            invoice_price,
            cust_name,
            gname
        FROM {table_name}
        WHERE gcode = '026436'
        ORDER BY create_dt DESC
        LIMIT 10
        """
        
        print("\n📋 执行的 SQL 查询:")
        print(query2)
        
        recent_data = pd.read_sql(query2, db_manager.engine)
        print("\n📊 最近的数据:")
        print(recent_data.to_string(index=False))
        
        # ========== 3. 查看日期分布 ==========
        print("\n\n[3] 查询 gcode=026436 的日期分布")
        print("-" * 80)
        
        query3 = f"""
        SELECT 
            YEAR(create_dt) as year,
            COUNT(*) as record_count,
            SUM(qty) as total_qty
        FROM {table_name}
        WHERE gcode = '026436'
        GROUP BY YEAR(create_dt)
        ORDER BY year DESC
        LIMIT 20
        """
        
        print("\n📋 执行的 SQL 查询:")
        print(query3)
        
        year_dist = pd.read_sql(query3, db_manager.engine)
        print("\n📊 按年份统计:")
        print(year_dist.to_string(index=False))
        
        # ========== 4. 查看客户分布 ==========
        print("\n\n[4] 查询 gcode=026436 的主要客户")
        print("-" * 80)
        
        query4 = f"""
        SELECT 
            cust_name,
            COUNT(*) as record_count,
            SUM(qty) as total_qty,
            MIN(create_dt) as first_date,
            MAX(create_dt) as last_date
        FROM {table_name}
        WHERE gcode = '026436'
        GROUP BY cust_name
        ORDER BY record_count DESC
        LIMIT 10
        """
        
        print("\n📋 执行的 SQL 查询:")
        print(query4)
        
        customer_dist = pd.read_sql(query4, db_manager.engine)
        print("\n📊 主要客户:")
        print(customer_dist.to_string(index=False))
        
        # ========== 5. 建议 ==========
        if len(result) > 0:
            total_count = result['total_count'].iloc[0]
            min_date = result['min_date'].iloc[0]
            max_date = result['max_date'].iloc[0]
            
            print("\n\n" + "=" * 80)
            print("✅ 数据检查完成！")
            print("=" * 80)
            print(f"\n📈 数据总览:")
            print(f"  - 总记录数: {total_count}")
            print(f"  - 日期范围: {min_date} 至 {max_date}")
            print(f"  - 唯一客户数: {result['unique_customers'].iloc[0]}")
            print(f"  - 总销量: {result['total_qty'].iloc[0]:.0f}")
            print(f"  - 平均销量: {result['avg_qty'].iloc[0]:.2f}")
            
            print(f"\n💡 建议:")
            if pd.to_datetime(min_date).year < 2000:
                print(f"  ⚠️ 检测到异常旧数据 ({min_date})，建议在查询时添加日期过滤")
                print(f"  ✓ 示例: start_date='2020-01-01', end_date='2024-12-31'")
            else:
                print(f"  ✓ 数据日期正常，可以直接使用")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 检查失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = check_query_and_data()
    sys.exit(0 if success else 1)
