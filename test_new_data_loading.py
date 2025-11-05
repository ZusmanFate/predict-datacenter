"""
测试新的数据加载功能
- 测试 dt 字段过滤（昨天）
- 测试近5年数据加载
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.utils.logger import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)

def test_data_loading():
    """测试新的数据加载功能"""
    
    print("\n" + "=" * 100)
    print(" 测试新的数据加载功能 ".center(100, "="))
    print("=" * 100)
    
    loader = DataLoader()
    gcode = "026436"
    
    # ========== 测试 1: 使用 dt=昨天 过滤 ==========
    print("\n[测试 1] 使用 dt=昨天 过滤")
    print("-" * 100)
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"昨天日期: {yesterday}")
    
    df1 = loader.load_sales_data(
        gcode=gcode,
        use_yesterday_dt=True  # 使用昨天作为 dt 过滤
    )
    
    print(f"\n结果:")
    print(f"  ✅ 加载 {len(df1)} 条记录")
    if len(df1) > 0:
        print(f"  日期范围: {df1['create_dt'].min()} 至 {df1['create_dt'].max()}")
        print(f"  客户数: {df1['cust_name'].nunique()}")
        print(f"  总销量: {df1['qty'].sum():,.0f}")
    
    # ========== 测试 2: 手动指定 dt 日期 ==========
    print("\n[测试 2] 手动指定 dt='2025-11-04'")
    print("-" * 100)
    
    df2 = loader.load_sales_data(
        gcode=gcode,
        dt_filter_date='2025-11-04'  # 手动指定 dt
    )
    
    print(f"\n结果:")
    print(f"  ✅ 加载 {len(df2)} 条记录")
    if len(df2) > 0:
        print(f"  日期范围: {df2['create_dt'].min()} 至 {df2['create_dt'].max()}")
        print(f"  客户数: {df2['cust_name'].nunique()}")
    
    # ========== 测试 3: 使用近5年数据 ==========
    print("\n[测试 3] 使用近5年数据")
    print("-" * 100)
    
    five_years_ago = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"自动设置日期范围: {five_years_ago} 至 {today}")
    
    df3 = loader.load_sales_data(
        gcode=gcode,
        use_last_5years=True  # 自动使用近5年数据
    )
    
    print(f"\n结果:")
    print(f"  ✅ 加载 {len(df3)} 条记录")
    if len(df3) > 0:
        print(f"  日期范围: {df3['create_dt'].min()} 至 {df3['create_dt'].max()}")
        print(f"  客户数: {df3['cust_name'].nunique()}")
        print(f"  总销量: {df3['qty'].sum():,.0f}")
    
    # ========== 测试 4: 手动指定日期范围（使用 dt 过滤）==========
    print("\n[测试 4] 手动指定日期范围 (2023-01-01 到 2024-12-31)")
    print("-" * 100)
    
    df4 = loader.load_sales_data(
        gcode=gcode,
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    
    print(f"\n结果:")
    print(f"  ✅ 加载 {len(df4)} 条记录")
    if len(df4) > 0:
        print(f"  日期范围: {df4['create_dt'].min()} 至 {df4['create_dt'].max()}")
        print(f"  客户数: {df4['cust_name'].nunique()}")
        print(f"  总销量: {df4['qty'].sum():,.0f}")
    
    # ========== 测试 5: 结合使用（近5年 + dt昨天）==========
    print("\n[测试 5] 结合使用: use_last_5years=True + use_yesterday_dt=True")
    print("-" * 100)
    print("注意: use_yesterday_dt 会覆盖 start_date/end_date 的 dt 过滤")
    
    df5 = loader.load_sales_data(
        gcode=gcode,
        use_last_5years=True,  # 设置 create_dt 范围为近5年
        use_yesterday_dt=True  # dt 过滤为昨天
    )
    
    print(f"\n结果:")
    print(f"  ✅ 加载 {len(df5)} 条记录")
    if len(df5) > 0:
        print(f"  日期范围: {df5['create_dt'].min()} 至 {df5['create_dt'].max()}")
        print(f"  说明: dt=昨天，但 create_dt 还是会在近5年范围内筛选")
    
    # ========== 测试 6: 检查新增字段 ==========
    print("\n[测试 6] 检查新增字段 (purchase_tax_rate, invoice_tax_rate)")
    print("-" * 100)
    
    df6 = loader.load_sales_data(gcode=gcode, limit=5)
    
    print(f"\n字段列表 ({len(df6.columns)} 个):")
    for i, col in enumerate(df6.columns, 1):
        marker = " ⭐" if col in ['purchase_tax_rate', 'invoice_tax_rate'] else ""
        print(f"  {i:2d}. {col}{marker}")
    
    if 'purchase_tax_rate' in df6.columns and 'invoice_tax_rate' in df6.columns:
        print(f"\n✅ 新增字段已成功包含!")
        print(f"\n数据样本:")
        print(df6[['gcode', 'qty', 'purchase_tax_rate', 'invoice_tax_rate']].head(3).to_string(index=False))
    
    # ========== 总结 ==========
    print("\n" + "=" * 100)
    print(" ✅ 测试完成！".center(100, "="))
    print("=" * 100)
    
    print("\n📊 数据量对比:")
    print(f"  测试1 (dt=昨天):        {len(df1):>8,} 条")
    print(f"  测试2 (dt=2025-11-04):  {len(df2):>8,} 条")
    print(f"  测试3 (近5年):          {len(df3):>8,} 条")
    print(f"  测试4 (2023-2024):      {len(df4):>8,} 条")
    print(f"  测试5 (近5年+dt昨天):   {len(df5):>8,} 条")
    
    print("\n💡 使用建议:")
    print("  1. 日常训练: use_last_5years=True  (使用近5年数据)")
    print("  2. 增量更新: use_yesterday_dt=True (只加载昨天的数据)")
    print("  3. 特定时间段: start_date='2023-01-01', end_date='2024-12-31'")
    print("  4. 性能优化: 优先使用 dt 过滤（分区列，查询更快）")

if __name__ == "__main__":
    test_data_loading()
