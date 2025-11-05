"""
Feast 特征定义
定义药品销量预测相关的特征
"""
from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from pathlib import Path

# 定义实体
drug_entity = Entity(
    name="drug",
    join_keys=["gcode"],
    description="药品实体"
)

hospital_entity = Entity(
    name="hospital", 
    join_keys=["cust_name"],
    description="医院（客户）实体"
)

# 数据源：销量数据
sales_source = FileSource(
    path=str(Path(__file__).parent.parent / "data/features/sales_features.parquet"),
    timestamp_field="create_dt",
)

# 特征视图：销量特征
sales_features = FeatureView(
    name="sales_features",
    entities=[drug_entity, hospital_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sales_quantity_lag_1", dtype=Float32),
        Field(name="sales_quantity_lag_7", dtype=Float32),
        Field(name="sales_quantity_lag_30", dtype=Float32),
        Field(name="sales_quantity_rolling_7_mean", dtype=Float32),
        Field(name="sales_quantity_rolling_14_mean", dtype=Float32),
        Field(name="sales_quantity_rolling_30_mean", dtype=Float32),
        Field(name="sales_quantity_rolling_7_std", dtype=Float32),
        Field(name="sales_quantity_rolling_30_std", dtype=Float32),
        Field(name="month", dtype=Int64),
        Field(name="dayofweek", dtype=Int64),
        Field(name="quarter", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
    ],
    source=sales_source,
    online=True,
)

# 特征视图：药品信息
drug_info_source = FileSource(
    path=str(Path(__file__).parent.parent / "data/features/drug_info.parquet"),
    timestamp_field="event_timestamp",
)

drug_info_features = FeatureView(
    name="drug_info_features",
    entities=[drug_entity],
    ttl=timedelta(days=3650),  # 药品信息变化较少
    schema=[
        Field(name="prod_cat1", dtype=String),
        Field(name="prod_cat2", dtype=String),
        Field(name="dosage_form", dtype=String),
        Field(name="holder_name", dtype=String),
        Field(name="pzwh", dtype=String),
    ],
    source=drug_info_source,
    online=True,
)
