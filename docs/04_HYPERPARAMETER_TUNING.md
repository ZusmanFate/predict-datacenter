# 🎯 超参数调优完整指南

## 三种主要方法

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 网格搜索 | 全面搜索 | 计算量大 | 参数空间小 |
| 随机搜索 | 速度快 | 可能错过最优 | 参数空间大 |
| **Optuna** ⭐ | 智能高效 | 需要额外依赖 | **任何场景** |

---

## 方法一：网格搜索（Grid Search）

### 适用场景
- 参数空间较小（<1000种组合）
- 需要全面搜索
- 计算资源充足

### 代码示例

```python
from sklearn.model_selection import GridSearchCV
from src.models.lgb_model import LightGBMModel

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 50]
}

# 创建模型
model = LightGBMModel()

# 网格搜索
grid_search = GridSearchCV(
    estimator=model.model,
    param_grid=param_grid,
    cv=5,  # 5折交叉验证
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # 使用所有CPU
    verbose=2
)

# 训练
print("开始网格搜索...")
grid_search.fit(X_train, y_train)

# 输出结果
print("\n最佳参数:")
print(grid_search.best_params_)
print(f"\n最佳得分: {-grid_search.best_score_:.2f}")

# 使用最佳参数的模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

### 完整脚本

```python
# scripts/grid_search_tune.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import GridSearchCV
from src.data.loader import DataLoader
from src.models.lgb_model import LightGBMModel
import pandas as pd

def grid_search_tuning(gcode, customer):
    # 加载数据
    # ... (数据加载和预处理)
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    
    # 网格搜索
    model = LightGBMModel()
    grid_search = GridSearchCV(
        model.model, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    # 保存结果
    results = {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_
    }
    
    return results

if __name__ == "__main__":
    results = grid_search_tuning("026436", "柳州市工人医院")
    print(results)
```

---

## 方法二：随机搜索（Random Search）

### 适用场景
- 参数空间大
- 想快速找到较好参数
- 时间有限

### 代码示例

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 定义参数分布
param_distributions = {
    'n_estimators': randint(50, 500),          # 50-500之间的整数
    'max_depth': randint(3, 10),               # 3-10之间的整数
    'learning_rate': uniform(0.01, 0.29),      # 0.01-0.3之间的浮点数
    'num_leaves': randint(20, 100),
    'min_child_samples': randint(10, 100),
    'subsample': uniform(0.6, 0.4),            # 0.6-1.0
    'colsample_bytree': uniform(0.6, 0.4)
}

# 随机搜索
model = LightGBMModel()
random_search = RandomizedSearchCV(
    estimator=model.model,
    param_distributions=param_distributions,
    n_iter=50,  # 尝试50次随机组合
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("开始随机搜索...")
random_search.fit(X_train, y_train)

print("\n最佳参数:")
print(random_search.best_params_)
```

---

## 方法三：Optuna 贝叶斯优化（推荐⭐）

### 为什么推荐 Optuna？

1. **智能搜索**: 基于贝叶斯优化，自动学习参数关系
2. **效率高**: 比网格搜索快10-100倍
3. **易于使用**: API简单直观
4. **功能强大**: 支持早停、剪枝、并行
5. **可视化**: 内置多种可视化工具

### 基础使用

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Optuna 优化目标函数"""
    
    # 定义搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }
    
    # 创建模型
    model = LightGBMModel()
    model.model.set_params(**params)
    
    # 交叉验证
    scores = cross_val_score(
        model.model, X_train, y_train,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # 返回平均得分
    return -scores.mean()

# 创建研究
study = optuna.create_study(
    direction='minimize',
    study_name='lightgbm_optimization'
)

# 运行优化
print("开始 Optuna 优化...")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# 输出结果
print("\n最佳参数:")
print(study.best_params)
print(f"\n最佳得分: {study.best_value:.2f}")
```

### 完整优化脚本

```python
# scripts/optuna_tune.py

import optuna
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.lgb_model import LightGBMModel
from sklearn.model_selection import cross_val_score
import json

def optimize_hyperparameters(gcode, customer, n_trials=100):
    """使用 Optuna 优化超参数"""
    
    print(f"优化参数: {gcode} - {customer}")
    
    # 1. 加载数据
    loader = DataLoader()
    df = loader.load_sales_data(gcode=gcode)
    df = df[df['cust_name'] == customer]
    
    # 2. 预处理和特征工程
    # ... (同 simple_demo_026436.py)
    
    # 3. 定义优化目标
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
        
        model = LightGBMModel()
        model.model.set_params(**params)
        
        scores = cross_val_score(
            model.model, X_train, y_train,
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        return -scores.mean()
    
    # 4. 运行优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 5. 保存结果
    best_params = study.best_params
    output_file = f"best_params_{gcode}_{customer[:20]}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'gcode': gcode,
            'customer': customer,
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': n_trials
        }, f, indent=2)
    
    print(f"\n✅ 优化完成!")
    print(f"最佳参数已保存到: {output_file}")
    print(f"最佳得分: {study.best_value:.2f}")
    
    return best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcode', default='026436')
    parser.add_argument('--customer', default='柳州市工人医院')
    parser.add_argument('--n-trials', type=int, default=100)
    args = parser.parse_args()
    
    optimize_hyperparameters(args.gcode, args.customer, args.n_trials)
```

### 运行脚本

```bash
# 基础运行
python scripts/optuna_tune.py --gcode 026436 --customer 柳州市工人医院

# 增加尝试次数
python scripts/optuna_tune.py --gcode 026436 --n-trials 200

# 批量优化多个组合
python scripts/batch_optuna_tune.py --config config/tune_list.yaml
```

### Optuna 高级功能

#### 1. 可视化优化过程

```python
import optuna.visualization as vis

# 优化历史
fig = vis.plot_optimization_history(study)
fig.show()
fig.write_html("optimization_history.html")

# 参数重要性
fig = vis.plot_param_importances(study)
fig.show()

# 参数关系
fig = vis.plot_parallel_coordinate(study)
fig.show()

# 切片图
fig = vis.plot_slice(study)
fig.show()
```

#### 2. 早停（Pruning）

```python
from optuna.pruners import MedianPruner

# 创建带早停的研究
study = optuna.create_study(
    direction='minimize',
    pruner=MedianPruner(n_startup_trials=10)
)

def objective_with_pruning(trial):
    params = {...}
    
    model = LightGBMModel()
    model.model.set_params(**params)
    
    # 逐折评估，允许早停
    for fold in range(5):
        score = ...  # 单折得分
        
        # 报告中间结果
        trial.report(score, fold)
        
        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return final_score

study.optimize(objective_with_pruning, n_trials=100)
```

#### 3. 并行优化

```python
# 方法1: 使用数据库后端（支持多进程）
study = optuna.create_study(
    storage='sqlite:///optuna.db',  # 共享数据库
    study_name='parallel_optimization',
    direction='minimize',
    load_if_exists=True
)

# 在多个终端/进程中运行
# 终端1: python optuna_tune.py
# 终端2: python optuna_tune.py
# 终端3: python optuna_tune.py
```

#### 4. 多目标优化

```python
def multi_objective(trial):
    params = {...}
    
    # 训练模型
    model = LightGBMModel()
    model.model.set_params(**params)
    model.fit(X_train, y_train)
    
    # 计算多个指标
    rmse = ...
    mae = ...
    
    return rmse, mae  # 返回多个目标

study = optuna.create_study(directions=['minimize', 'minimize'])
study.optimize(multi_objective, n_trials=100)
```

---

## 使用最佳参数

### 加载并应用参数

```python
import json

# 加载最佳参数
with open('best_params_026436_柳州市工人医院.json', 'r') as f:
    result = json.load(f)
    best_params = result['best_params']

# 创建模型
model = LightGBMModel()
model.model.set_params(**best_params)

# 训练
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
```

---

## 调优策略建议

### 阶段1: 快速探索（20-50次）
```python
# 使用随机搜索或 Optuna
# 目标：找到大致的参数范围
n_trials = 50
```

### 阶段2: 精细优化（100-200次）
```python
# 使用 Optuna
# 基于第一阶段的结果，缩小搜索范围
n_trials = 200
```

### 阶段3: 最终调整（手动）
```python
# 基于最佳参数，手动微调
# 观察训练/验证曲线
```

---

## 常见问题

### Q1: 优化需要多长时间？
- 网格搜索：几小时到几天
- 随机搜索：30分钟-2小时
- **Optuna**: 1-3小时（100次）

### Q2: 需要多少次尝试？
- 快速：20-50次
- 标准：100-200次
- 深度：300-500次

### Q3: 如何避免过拟合？
- 使用交叉验证（cv=5）
- 监控验证集性能
- 使用早停
- 适当的正则化

### Q4: 可以中断后继续吗？
- Optuna: ✅ 支持（使用数据库后端）
- 网格/随机搜索: ❌ 需要重新开始
