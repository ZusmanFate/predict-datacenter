# 🤖 LightGBM 模型参数详解

## 默认参数配置

查看文件：`src/models/lgb_model.py`

```python
self.model_params = {
    'objective': 'regression',        # 回归任务
    'metric': 'rmse',                 # 评估指标
    'n_estimators': 100,              # 树的数量
    'max_depth': 6,                   # 树的最大深度
    'learning_rate': 0.1,             # 学习率
    'num_leaves': 31,                 # 叶子节点数
    'min_child_samples': 20,          # 叶子最小样本数
    'subsample': 0.8,                 # 数据采样比例
    'colsample_bytree': 0.8,          # 特征采样比例
    'random_state': 42,               # 随机种子
    'n_jobs': -1,                     # 并行线程数
    'verbose': -1                     # 不打印训练日志
}
```

## 核心参数详解

### 1. n_estimators（树的数量）
- **默认值**: 100
- **取值范围**: 50-500
- **说明**: 模型中树的总数，越多模型越复杂
- **影响**: 
  - 增加：提升性能，但可能过拟合，训练时间更长
  - 减少：训练快，但可能欠拟合
- **调优建议**: 
  - 先试 [50, 100, 200]
  - 结合 learning_rate 调整
  - 观察验证集性能，避免过拟合

### 2. max_depth（树的最大深度）
- **默认值**: 6
- **取值范围**: 3-10
- **说明**: 每棵树的最大深度
- **影响**:
  - 增加：捕捉更复杂的模式，但易过拟合
  - 减少：模型简单，泛化能力强
- **调优建议**:
  - 数据量大：6-8
  - 数据量小：3-5
  - 先从默认值开始

### 3. learning_rate（学习率）
- **默认值**: 0.1
- **取值范围**: 0.01-0.3
- **说明**: 每棵树对最终结果的贡献程度
- **影响**:
  - 降低：需要更多树，但模型更稳定
  - 提高：训练快，但可能不稳定
- **调优建议**:
  - 如果过拟合：降低到 0.05 或 0.01
  - 配合 n_estimators 调整
  - 常用值：0.01, 0.05, 0.1

### 4. num_leaves（叶子节点数）
- **默认值**: 31
- **取值范围**: 20-100
- **说明**: 每棵树的最大叶子数
- **影响**: 控制模型复杂度
- **调优建议**:
  - 建议 < 2^max_depth
  - max_depth=6 时，31 是合理值
  - 过拟合时减小

### 5. min_child_samples（叶子最小样本数）
- **默认值**: 20
- **取值范围**: 10-100
- **说明**: 叶子节点包含的最小样本数
- **影响**: 防止过拟合
- **调优建议**:
  - 数据量大：可以增加到 50-100
  - 数据量小：保持 20
  - 过拟合时增加

### 6. subsample（数据采样比例）
- **默认值**: 0.8
- **取值范围**: 0.6-1.0
- **说明**: 每次迭代使用的数据比例
- **影响**: 防止过拟合，增加随机性
- **调优建议**:
  - 过拟合：降低到 0.6-0.7
  - 欠拟合：增加到 0.9-1.0

### 7. colsample_bytree（特征采样比例）
- **默认值**: 0.8
- **取值范围**: 0.6-1.0
- **说明**: 每棵树使用的特征比例
- **影响**: 防止过拟合，特征随机性
- **调优建议**:
  - 特征很多：降低到 0.6-0.7
  - 特征较少：保持 0.8-1.0

## 正则化参数

### reg_alpha（L1正则化）
- **默认值**: 0
- **取值范围**: 0-10
- **用途**: 特征选择，使部分权重为0
- **调优建议**: [0, 0.1, 1, 10]

### reg_lambda（L2正则化）
- **默认值**: 0
- **取值范围**: 0-10
- **用途**: 权重平滑，防止过拟合
- **调优建议**: [0, 0.1, 1, 10]

## 如何修改参数

### 方法1: 直接修改模型文件
```python
# 编辑: src/models/lgb_model.py
class LightGBMModel(BaseModel):
    def __init__(self):
        self.model_params = {
            'n_estimators': 200,      # 修改为200
            'learning_rate': 0.05,    # 修改为0.05
            'max_depth': 8,           # 修改为8
            # ... 其他参数
        }
```

### 方法2: 训练时传入参数
```python
from src.models.lgb_model import LightGBMModel

# 创建模型
model = LightGBMModel()

# 自定义参数
custom_params = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 8
}

# 应用参数
model.model.set_params(**custom_params)

# 训练
model.fit(X_train, y_train)
```

### 方法3: 使用配置文件
```yaml
# config/model_config.yaml
lightgbm:
  n_estimators: 200
  learning_rate: 0.05
  max_depth: 8
  num_leaves: 50
  min_child_samples: 30
  subsample: 0.7
  colsample_bytree: 0.7
```

## 参数调优策略

### 1. 粗调（快速找到大致范围）
```python
params_coarse = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1]
}
```

### 2. 细调（在最佳范围内精调）
```python
params_fine = {
    'n_estimators': [80, 100, 120],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.05, 0.08, 0.1],
    'num_leaves': [25, 31, 40]
}
```

### 3. 正则化调整
```python
params_regularization = {
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}
```

## 常见问题诊断

### 问题1: 过拟合（训练好，测试差）
**解决方案**:
- 减小 max_depth: 6 → 4
- 减小 num_leaves: 31 → 20
- 增加 min_child_samples: 20 → 50
- 减小 subsample: 0.8 → 0.6
- 减小 colsample_bytree: 0.8 → 0.6
- 增加正则化: reg_lambda = 1

### 问题2: 欠拟合（训练差，测试也差）
**解决方案**:
- 增加 n_estimators: 100 → 200
- 增加 max_depth: 6 → 8
- 增加 num_leaves: 31 → 50
- 增加 learning_rate: 0.1 → 0.2

### 问题3: 训练太慢
**解决方案**:
- 减少 n_estimators: 100 → 50
- 增加 learning_rate: 0.1 → 0.2
- 减小 max_depth: 6 → 4
- 使用更多 CPU: n_jobs = -1

## 参数组合示例

### 保守配置（防止过拟合）
```python
conservative_params = {
    'n_estimators': 50,
    'max_depth': 4,
    'learning_rate': 0.05,
    'num_leaves': 20,
    'min_child_samples': 50,
    'subsample': 0.6,
    'colsample_bytree': 0.6
}
```

### 激进配置（追求性能）
```python
aggressive_params = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.1,
    'num_leaves': 80,
    'min_child_samples': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.9
}
```

### 平衡配置（推荐起点）
```python
balanced_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```
