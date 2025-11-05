"""
测试新增字段是否能正常查询
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader

def test_new_fields():
    """测试新字段"""
    print("\n测试新字段查询...")
    
    loader = DataLoader()
    
    try:
        # 简单查询测试
        df = loader.load_sales_data(gcode="026436", limit=5)
        
        print(f"✅ 成功加载 {len(df)} 条数据")
        print(f"\n字段列表 ({len(df.columns)} 个):")
        for i, col in enumerate(df.columns, 1):
            marker = " ⭐" if col in ['purchase_tax_rate', 'invoice_tax_rate'] else ""
            print(f"  {i:2d}. {col}{marker}")
        
        if 'purchase_tax_rate' in df.columns and 'invoice_tax_rate' in df.columns:
            print(f"\n✅ 新增字段成功包含！")
            print(f"\n数据样本:")
            print(df[['gcode', 'qty', 'purchase_tax_rate', 'invoice_tax_rate']].to_string(index=False))
        else:
            print(f"\n⚠️ 缺少新字段")
            
    except Exception as e:
        print(f"❌ 查询失败: {e}")

if __name__ == "__main__":
    test_new_fields()
