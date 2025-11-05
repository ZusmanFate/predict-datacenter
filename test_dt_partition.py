"""
测试 dt 分区字段和 create_dt 的正确使用
- dt: 分区字段，表示数据采集日期（按天全量）
- create_dt: 开单日期（业务日期）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.utils.logger import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)

def test_dt_partition():
    """测试 dt 分区字段"""
    
    print("\n" + "=" * 100)
    print(" 测试 dt 分区字段和 create_dt 筛选 ".center(100, "="))
    print("=" * 100)
    
    loader = DataLoader()
    gcode = "026436"
    
    # ========== 测试 1: 使用 dt=昨天（从昨天的全量分区读数据）==========
    print("\n[测试 1] dt=昨天（从昨天的全量分区读数据）")
    print("-" * 100)
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"昨天日期: {yesterday}")
    print(f"说明: dt='{yesterday}' 表示从 {yesterday} 这天采集的全量数据中读取")
    
    try:
        df1 = loader.load_sales_data(
            gcode=gcode,
            use_yesterday_dt=True  # dt=昨天（分区过滤）
        )
        
        print(f"\n✅ 成功加载 {len(df1)} 条记录")
        if len(df1) > 0:
            print(f"   开单日期范围 (create_dt): {df1['create_dt'].min()} 至 {df1['create_dt'].max()}")
            print(f"   客户数: {df1['cust_name'].nunique()}")
            print(f"   总销量: {df1['qty'].sum():,.0f}")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # ========== 测试 2: dt=昨天 + create_dt 筛选近5年 ==========
    print("\n[测试 2] dt=昨天 + create_dt 筛选近5年")
    print("-" * 100)
    
    five_years_ago = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"dt: {yesterday} (从昨天的全量分区读数据)")
    print(f"create_dt 范围: {five_years_ago} 至 {today}")
    
    try:
        df2 = loader.load_sales_data(
            gcode=gcode,
            use_yesterday_dt=True,  # dt=昨天
            use_last_5years=True    # create_dt 筛选近5年
        )
        
        print(f"\n✅ 成功加载 {len(df2)} 条记录")
        if len(df2) > 0:
            print(f"   开单日期范围 (create_dt): {df2['create_dt'].min()} 至 {df2['create_dt'].max()}")
            print(f"   说明: 数据来自昨天的全量分区，但开单日期在近5年内")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # ========== 测试 3: 手动指定 dt + create_dt 范围 ==========
    print("\n[测试 3] 手动指定 dt='2025-11-04' + create_dt 范围")
    print("-" * 100)
    
    print(f"dt: 2025-11-04 (从这天的全量分区读数据)")
    print(f"create_dt: 2023-01-01 至 2024-12-31")
    
    try:
        df3 = loader.load_sales_data(
            gcode=gcode,
            dt_filter_date='2025-11-04',  # dt=指定日期
            start_date='2023-01-01',      # create_dt >= 2023-01-01
            end_date='2024-12-31'         # create_dt <= 2024-12-31
        )
        
        print(f"\n✅ 成功加载 {len(df3)} 条记录")
        if len(df3) > 0:
            print(f"   开单日期范围 (create_dt): {df3['create_dt'].min()} 至 {df3['create_dt'].max()}")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # ========== 测试 4: 只用 create_dt 筛选（不指定 dt）==========
    print("\n[测试 4] 只用 create_dt 筛选（不指定 dt 分区）")
    print("-" * 100)
    
    print(f"dt: 未指定（扫描所有分区）")
    print(f"create_dt: 2024-01-01 至 2024-12-31")
    
    try:
        df4 = loader.load_sales_data(
            gcode=gcode,
            start_date='2024-01-01',
            end_date='2024-12-31',
            limit=1000  # 限制1000条
        )
        
        print(f"\n✅ 成功加载 {len(df4)} 条记录 (limit=1000)")
        if len(df4) > 0:
            print(f"   开单日期范围 (create_dt): {df4['create_dt'].min()} 至 {df4['create_dt'].max()}")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # ========== 测试 5: 只指定 dt，不限制 create_dt ==========
    print("\n[测试 5] 只指定 dt=昨天，不限制 create_dt")
    print("-" * 100)
    
    print(f"dt: {yesterday}")
    print(f"create_dt: 不限制（全部开单日期）")
    
    try:
        df5 = loader.load_sales_data(
            gcode=gcode,
            dt_filter_date=yesterday  # 只指定 dt
        )
        
        print(f"\n✅ 成功加载 {len(df5)} 条记录")
        if len(df5) > 0:
            print(f"   开单日期范围 (create_dt): {df5['create_dt'].min()} 至 {df5['create_dt'].max()}")
            print(f"   说明: 这些数据都是从 dt={yesterday} 这天的全量分区读取的")
            print(f"   但开单日期 create_dt 可能跨越多年（历史订单）")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # ========== 总结 ==========
    print("\n" + "=" * 100)
    print(" 测试总结 ".center(100, "="))
    print("=" * 100)
    
    print("\n📋 字段说明:")
    print("  • dt:        分区字段，数据采集日期（按天全量），格式 YYYY-MM-DD")
    print("               例如 dt='2025-11-04' 表示从昨天采集的全量数据中读取")
    print("  • create_dt: 开单日期，实际销售发生的日期（业务日期）")
    print("               start_date/end_date 用于筛选 create_dt")
    
    print("\n💡 使用建议:")
    print("  1. 日常增量训练:")
    print("     use_yesterday_dt=True, use_last_5years=True")
    print("     → 从昨天的全量分区读取，筛选开单日期在近5年的数据")
    
    print("\n  2. 特定时间段训练:")
    print("     dt_filter_date='2025-11-04', start_date='2023-01-01', end_date='2024-12-31'")
    print("     → 从指定分区读取，筛选指定开单日期范围")
    
    print("\n  3. 性能优化:")
    print("     ✅ 指定 dt 可以大幅提升查询性能（分区剪枝）")
    print("     ✅ 建议日常使用 use_yesterday_dt=True 读取最新全量数据")

if __name__ == "__main__":
    test_dt_partition()
