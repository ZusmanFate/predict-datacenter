# 📊 数据库字段说明

## 数据表信息
- **表名**: `datasense_dlink_erpservice.view_dws_erp_sal_detail_df`
- **数据库**: StarRocks
- **字段数量**: 19 个

## 字段列表

| # | 字段名 | 类型 | 说明 | 用途 |
|---|--------|------|------|------|
| 1 | **gcode** | VARCHAR | 药品编码 | 🔑 筛选特定药品（如 026436） |
| 2 | **create_dt** | DATE | 销售日期 | 🔑 时间序列关键字段 |
| 3 | **qty** | DECIMAL | 销售数量 | 🎯 **预测目标** |
| 4 | **invoice_price** | DECIMAL | 开票价格 | 💰 价格信息 |
| 5 | **cust_name** | VARCHAR | 客户名称 | 🏥 区分不同医院/客户 |
| 6 | **code** | VARCHAR | 产品代码 | 📦 产品标识 |
| 7 | **gname** | VARCHAR | 药品名称 | 📝 药品描述 |
| 8 | **mfr_custno** | VARCHAR | 生产厂家编号 | 🏭 厂家信息 |
| 9 | **mfr_name** | VARCHAR | 生产厂家名称 | 🏭 厂家名称 |
| 10 | **pack_l** | DECIMAL | 包装规格（大） | 📦 包装信息 |
| 11 | **pack_m** | DECIMAL | 包装规格（中） | 📦 包装信息 |
| 12 | **purchase_price** | DECIMAL | 采购价格 | 💵 成本信息 |
| 13 | **invoice_wholesale_price** | DECIMAL | 批发开票价 | 💰 批发价格 |
| 14 | **whs_attr_code** | VARCHAR | 仓库属性代码 | 🏪 仓库信息 |
| 15 | **pzwh** | VARCHAR | 批准文号 | 📋 药品许可 |
| 16 | **prod_dt** | VARCHAR | 生产日期 | 📅 生产信息 |
| 17 | **valid_dt** | VARCHAR | 有效期至 | 📅 有效期信息 |
| 18 | **sale_area_id** | VARCHAR | 销售区域ID | 🗺️ 区域信息 |
| 19 | **contno** | VARCHAR | 合同编号 | 📄 合同信息 |

## 数据样本（gcode=026436）
```
gcode   create_dt   qty   cust_name                  gname
026436  2025-11-03  64.0  广西壮族自治区人民医院      低钙腹膜透析液
026436  2025-11-03  24.0  广西医科大学第一附属医院    低钙腹膜透析液
026436  2025-11-03  136.0 广西中医药大学第一附属医院  低钙腹膜透析液
```

## 查询示例

```python
from src.data.loader import DataLoader

loader = DataLoader()

# 加载特定药品数据
df = loader.load_sales_data(gcode="026436")

# 查看字段
print(df.columns.tolist())
print(df.dtypes)
print(df.head())
```

## SQL 查询语句

```sql
SELECT 
    gcode, create_dt, qty, cust_name, gname, 
    invoice_price, code, mfr_name, ...
FROM datasense_dlink_erpservice.view_dws_erp_sal_detail_df
WHERE gcode = '026436'
  AND create_dt >= '2020-01-01'
ORDER BY create_dt ASC
```
