"""
检查数据库中是否存在新增字段
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.database import get_db_manager
import pandas as pd

def check_fields():
    """检查字段"""
    print("\n检查数据库表字段...")
    
    db_manager = get_db_manager()
    table_name = "datasense_dlink_erpservice.view_dws_erp_sal_detail_df"
    
    # 方法1: 查询表结构
    print(f"\n[方法1] 查询表结构:")
    try:
        query = f"DESC {table_name}"
        df = pd.read_sql(query, db_manager.engine)
        print(df.to_string(index=False))
        
        # 检查新字段
        field_names = df['Field'].tolist() if 'Field' in df.columns else df.iloc[:, 0].tolist()
        
        print(f"\n✅ 表中共有 {len(field_names)} 个字段")
        
        if 'purchase_tax_rate' in field_names:
            print("✅ purchase_tax_rate 字段存在")
        else:
            print("❌ purchase_tax_rate 字段不存在")
        
        if 'invoice_tax_rate' in field_names:
            print("✅ invoice_tax_rate 字段存在")
        else:
            print("❌ invoice_tax_rate 字段不存在")
            
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    
    # 方法2: 尝试查询前几行数据（不包含新字段）
    print(f"\n[方法2] 查询前3行数据（基础字段）:")
    try:
        query = f"""
        SELECT gcode, create_dt, qty, cust_name, invoice_price
        FROM {table_name}
        WHERE gcode = '026436'
        LIMIT 3
        """
        df = pd.read_sql(query, db_manager.engine)
        print(df.to_string(index=False))
        print(f"\n✅ 基础字段查询成功")
    except Exception as e:
        print(f"❌ 查询失败: {e}")

if __name__ == "__main__":
    check_fields()
