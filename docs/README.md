# 📚 完整文档导航

欢迎查看药品销量预测系统的完整文档！

## 📋 文档列表

### 1️⃣ [数据库字段说明](01_DATABASE_FIELDS.md)
**内容**:
- 数据表结构（19个字段）
- 字段类型和说明
- 数据样本展示
- SQL 查询示例

**关键信息**:
- 表名: `datasense_dlink_erpservice.view_dws_erp_sal_detail_df`
- 关键字段: `gcode`(药品编码), `create_dt`(日期), `qty`(销量), `cust_name`(客户)
- 数据量: 177,051 个药品, 19,011 个客户

---

### 2️⃣ [特征工程详解](02_FEATURES.md)
**内容**:
- 49个特征的完整列表
- 特征分类（日期、滞后、滚动窗口、差分、统计）
- 特征重要性排名
- 如何查看特征

**关键发现**:
- 最重要特征: `sales_quantity_diff_7` (7天差分)
- 差分特征 > 滞后特征 > 滚动窗口特征
- 星期效应显著（`dayofweek`）

---

### 3️⃣ [模型参数详解](03_MODEL_PARAMS.md)
**内容**:
- LightGBM 默认参数配置
- 12个核心参数详解
- 参数调优建议
- 3种修改参数的方法
- 常见问题诊断

**核心参数**:
```python
{
    'n_estimators': 100,      # 树的数量
    'max_depth': 6,           # 树的深度
    'learning_rate': 0.1,     # 学习率
    'num_leaves': 31,         # 叶子节点数
    'min_child_samples': 20   # 叶子最小样本数
}
```

---

### 4️⃣ [超参数调优指南](04_HYPERPARAMETER_TUNING.md)
**内容**:
- 3种调优方法对比（网格搜索、随机搜索、Optuna）
- 完整代码示例
- Optuna 高级功能（可视化、早停、并行）
- 调优策略和最佳实践

**推荐方法**: **Optuna 贝叶斯优化** ⭐
- 智能高效，比网格搜索快10-100倍
- 支持可视化和并行
- 典型用法: 100-200次尝试，1-3小时

**示例**:
```bash
python scripts/optuna_tune.py --gcode 026436 --n-trials 100
```

---

### 5️⃣ [批量建模指南](05_BATCH_TRAINING.md)
**内容**:
- 4种批量训练方法
- 并行训练（推荐，4-8个进程）
- 智能自动选择组合
- 使用配置文件管理
- 批量预测和模型管理

**推荐方法**: **并行训练 + 自动选择** ⭐
```bash
# 自动选择并并行训练
python scripts/auto_batch_train.py \
    --min-records 500 \
    --top-products 20 \
    --top-customers 10 \
    --parallel 4
```

---

## 🎯 快速查找

### 我想知道...

| 问题 | 查看文档 |
|------|----------|
| 数据库有哪些字段？ | [01_DATABASE_FIELDS.md](01_DATABASE_FIELDS.md) |
| 特征是怎么生成的？ | [02_FEATURES.md](02_FEATURES.md) |
| 哪些特征最重要？ | [02_FEATURES.md](02_FEATURES.md) #特征重要性排名 |
| 模型参数是什么？ | [03_MODEL_PARAMS.md](03_MODEL_PARAMS.md) |
| 如何调整参数？ | [03_MODEL_PARAMS.md](03_MODEL_PARAMS.md) #如何修改参数 |
| 如何优化超参数？ | [04_HYPERPARAMETER_TUNING.md](04_HYPERPARAMETER_TUNING.md) |
| 如何批量训练模型？ | [05_BATCH_TRAINING.md](05_BATCH_TRAINING.md) |
| 如何并行加速？ | [05_BATCH_TRAINING.md](05_BATCH_TRAINING.md) #方法二 |

---

## 📊 数据概览

### 数据库原始字段（19个）
```
1. gcode           - 药品编码（筛选条件）
2. create_dt       - 销售日期（时间序列）
3. qty             - 销量（预测目标）⭐
4. invoice_price   - 价格
5. cust_name       - 客户名称（区分客户）
6-19. 其他字段     - 产品信息、厂家、包装等
```

### 特征工程产物（49个特征）
```
日期特征     14个 (year, month, day, dayofweek, ...)
滞后特征      6个 (lag_1, lag_2, lag_3, lag_7, lag_14, lag_30)
滚动窗口特征  20个 (rolling_3/7/14/30_mean/std/min/max/median)
差分特征      2个 (diff_1, diff_7)
统计特征      3个 (cumsum, cummean, cumstd)
```

### 模型参数（12个核心参数）
```
树结构:  n_estimators, max_depth, num_leaves
学习:    learning_rate, min_child_samples
正则化:  subsample, colsample_bytree, reg_alpha, reg_lambda
其他:    objective, metric, random_state
```

---

## 🚀 实战示例

### 示例1: 查看某个药品的数据
```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_sales_data(gcode="026436")

print(f"数据量: {len(df)}")
print(f"字段: {df.columns.tolist()}")
print(f"客户数: {df['cust_name'].nunique()}")
```

### 示例2: 查看特征
```python
from src.features.builder import FeatureBuilder

builder = FeatureBuilder()
df_features = builder.build_features(df, target_column='sales_quantity')

print(f"特征数: {len(df_features.columns)}")
print("特征列表:")
for col in df_features.columns:
    print(f"  - {col}")
```

### 示例3: 查看模型参数
```python
from src.models.lgb_model import LightGBMModel

model = LightGBMModel()
print("模型参数:")
for key, value in model.model_params.items():
    print(f"  {key}: {value}")
```

### 示例4: 优化超参数
```bash
# 运行 Optuna 优化
python scripts/optuna_tune.py \
    --gcode 026436 \
    --customer 柳州市工人医院 \
    --n-trials 100
```

### 示例5: 批量训练
```bash
# 自动选择并批量训练
python scripts/auto_batch_train.py \
    --min-records 500 \
    --top-products 20 \
    --top-customers 10 \
    --parallel 4
```

---

## 💡 常见问题 FAQ

### Q1: 数据库字段太多，我应该关注哪些？
**A**: 重点关注：
- `gcode` - 筛选药品
- `create_dt` - 时间序列
- `qty` - 预测目标
- `cust_name` - 区分客户

### Q2: 为什么特征有49个这么多？
**A**: 时间序列预测需要捕捉多种模式：
- 历史趋势（滞后特征）
- 周期性（日期特征）
- 波动性（滚动窗口）
- 变化率（差分特征）

### Q3: 模型参数应该怎么调？
**A**: 三步走：
1. 先用默认参数训练
2. 用 Optuna 自动优化（推荐）
3. 根据结果手动微调

### Q4: 超参数调优要多久？
**A**: 
- 快速：20-50次，30分钟-1小时
- 标准：100-200次，1-3小时
- 深度：300-500次，3-6小时

### Q5: 批量建模需要多少时间？
**A**:
- 单个模型：1-5分钟
- 100个模型（并行4进程）：30-60分钟
- 1000个模型（并行8进程）：3-6小时

---

## 📁 相关文件位置

### 配置文件
- `config/database.yaml` - 数据库配置
- `config/model_config.yaml` - 模型配置
- `config/batch_config.yaml` - 批量训练配置

### 脚本文件
- `simple_demo_026436.py` - 完整工作流示例
- `scripts/optuna_tune.py` - 超参数优化
- `scripts/auto_batch_train.py` - 批量训练
- `scripts/batch_predict.py` - 批量预测

### 数据文件
- `logs/app.log` - 运行日志
- `models/batch/` - 批量训练的模型
- `reports/` - 训练报告

---

## 🎓 学习路径建议

### 新手入门
1. 阅读 [01_DATABASE_FIELDS.md](01_DATABASE_FIELDS.md) - 了解数据
2. 运行 `simple_demo_026436.py` - 体验完整流程
3. 阅读 [02_FEATURES.md](02_FEATURES.md) - 理解特征

### 进阶使用
4. 阅读 [03_MODEL_PARAMS.md](03_MODEL_PARAMS.md) - 学习参数
5. 尝试修改参数，观察效果
6. 阅读 [04_HYPERPARAMETER_TUNING.md](04_HYPERPARAMETER_TUNING.md) - 优化模型

### 高级应用
7. 阅读 [05_BATCH_TRAINING.md](05_BATCH_TRAINING.md) - 批量建模
8. 实践批量训练和预测
9. 部署 API 服务（参考主文档）

---

## 🔗 相关链接

- [项目主文档](../README.md)
- [运行指南](../RUN_GUIDE.md)
- [工作流程总结](../WORKFLOW_SUMMARY.md)

---

**文档更新时间**: 2025-11-05

**如有问题，请查看日志**: `logs/app.log`
