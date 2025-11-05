# 🔧 特征工程完整说明

## 特征总览

**总特征数**: 49 个
**分类**: 日期特征、滞后特征、滚动窗口、差分特征、统计特征

## 1️⃣ 日期特征（14个）

### 基础日期特征（10个）

| 特征名 | 说明 | 示例值 | 用途 |
|--------|------|--------|------|
| year | 年份 | 2024 | 捕捉年度趋势 |
| month | 月份 | 1-12 | 捕捉季节性 |
| day | 日 | 1-31 | 日期效应 |
| dayofweek | 星期几 | 0-6 | 周期性模式 |
| dayofyear | 一年中第几天 | 1-366 | 年内位置 |
| weekofyear | 一年中第几周 | 1-53 | 周期性 |
| quarter | 季度 | 1-4 | 季度效应 |
| is_weekend | 是否周末 | 0/1 | 周末效应 |
| is_month_start | 是否月初 | 0/1 | 月初效应 |
| is_month_end | 是否月末 | 0/1 | 月末效应 |

### 周期性特征（4个）

| 特征名 | 说明 | 取值范围 |
|--------|------|----------|
| month_sin | 月份正弦编码 | -1 ~ 1 |
| month_cos | 月份余弦编码 | -1 ~ 1 |
| dayofweek_sin | 星期正弦编码 | -1 ~ 1 |
| dayofweek_cos | 星期余弦编码 | -1 ~ 1 |

## 2️⃣ 滞后特征（6个）

| 特征名 | 说明 |
|--------|------|
| sales_quantity_lag_1 | 1天前销量 |
| sales_quantity_lag_2 | 2天前销量 |
| sales_quantity_lag_3 | 3天前销量 |
| sales_quantity_lag_7 | 7天前销量 |
| sales_quantity_lag_14 | 14天前销量 |
| sales_quantity_lag_30 | 30天前销量 |

## 3️⃣ 滚动窗口特征（20个）

窗口大小：3天、7天、14天、30天
统计量：mean、std、min、max、median

完整列表：
```
sales_quantity_rolling_3_mean
sales_quantity_rolling_3_std
sales_quantity_rolling_3_min
sales_quantity_rolling_3_max
sales_quantity_rolling_3_median

sales_quantity_rolling_7_mean
sales_quantity_rolling_7_std
sales_quantity_rolling_7_min
sales_quantity_rolling_7_max
sales_quantity_rolling_7_median

sales_quantity_rolling_14_mean
sales_quantity_rolling_14_std
sales_quantity_rolling_14_min
sales_quantity_rolling_14_max
sales_quantity_rolling_14_median

sales_quantity_rolling_30_mean
sales_quantity_rolling_30_std
sales_quantity_rolling_30_min
sales_quantity_rolling_30_max
sales_quantity_rolling_30_median
```

## 4️⃣ 差分特征（2个）

| 特征名 | 计算公式 |
|--------|----------|
| sales_quantity_diff_1 | 今天 - 昨天 |
| sales_quantity_diff_7 | 今天 - 7天前 |

## 5️⃣ 统计特征（3个）

| 特征名 | 说明 |
|--------|------|
| sales_quantity_cumsum | 累计总销量 |
| sales_quantity_cummean | 累计平均销量 |
| sales_quantity_cumstd | 累计标准差 |

## 特征重要性排名（实际运行结果）

| 排名 | 特征名 | 重要性 | 类型 |
|------|--------|--------|------|
| 1 | sales_quantity_diff_7 | 443,946,984 | 差分 |
| 2 | sales_quantity_diff_1 | 302,854,074 | 差分 |
| 3 | sales_quantity_lag_1 | 164,206,305 | 滞后 |
| 4 | sales_quantity_lag_7 | 92,669,345 | 滞后 |
| 5 | sales_quantity_rolling_7_max | 62,388,930 | 滚动 |
| 6 | sales_quantity_rolling_7_min | 27,604,271 | 滚动 |
| 7 | sales_quantity_rolling_7_mean | 7,613,630 | 滚动 |
| 8 | sales_quantity_rolling_7_std | 5,082,037 | 滚动 |
| 9 | sales_quantity_rolling_7_median | 4,773,031 | 滚动 |
| 10 | dayofweek | 4,168,662 | 日期 |

## 如何查看特征

```python
from src.features.builder import FeatureBuilder

# 构建特征
builder = FeatureBuilder()
df_features = builder.build_features(df, target_column='sales_quantity')

# 查看所有特征
print("特征列表:")
print(df_features.columns.tolist())

# 查看特征统计
print("\n特征统计:")
print(df_features.describe())
```
