# 🚀 批量建模完整指南

## 为什么需要批量建模？

### 业务场景
- **多药品**: 177,051 个唯一药品
- **多客户**: 19,011 个唯一客户
- **个性化**: 不同药品-客户组合有不同的销售模式

### 单模型 vs 批量模型

| 方式 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| 单模型 | 所有数据训练一个模型 | 简单 | 预测不准确 |
| **批量模型** | 每个组合一个模型 | 个性化，精准 | 需要管理多个模型 |

---

## 方法一：简单循环批量训练

### 适用场景
- 组合数量较少（<100个）
- 不需要并行
- 代码简单易懂

### 完整代码

```python
# scripts/simple_batch_train.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.models.lgb_model import LightGBMModel
from src.training.trainer import ModelTrainer
import pandas as pd
import json

def train_single(gcode, customer):
    """训练单个药品-客户组合"""
    try:
        print(f"\n训练: {gcode} - {customer}")
        
        # 1. 加载数据
        loader = DataLoader()
        df = loader.load_sales_data(gcode=gcode)
        df = df[df['cust_name'] == customer]
        
        # 检查数据量
        if len(df) < 100:
            print(f"  ⚠️  跳过: 数据不足 ({len(df)} 条)")
            return {
                'gcode': gcode,
                'customer': customer,
                'status': 'skipped',
                'reason': 'insufficient_data',
                'record_count': len(df)
            }
        
        # 2. 预处理
        processor = DataProcessor()
        df_proc = df.rename(columns={
            'create_dt': 'date',
            'qty': 'sales_quantity',
            'gcode': 'drug_id',
            'cust_name': 'hospital_id'
        })
        df_proc['date'] = pd.to_datetime(df_proc['date'])
        df_proc = df_proc.groupby(['date', 'drug_id', 'hospital_id']).agg({
            'sales_quantity': 'sum'
        }).reset_index().sort_values('date')
        
        df_proc = processor.create_time_series_dataset(
            df_proc, drug_id=gcode, hospital_id=customer,
            date_column='date', target_column='sales_quantity'
        )
        df_proc = processor.handle_missing_values(df_proc, method='forward_fill')
        df_proc = processor.handle_outliers(df_proc, 'sales_quantity', method='iqr')
        
        # 3. 特征工程
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(
            df_proc, target_column='sales_quantity', date_column='date'
        )
        
        # 4. 准备训练数据
        exclude_cols = {'sales_quantity', 'date', 'drug_id', 'hospital_id'}
        numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols 
                       and df_features[col].dtype.name in numeric_types]
        
        X = df_features[feature_cols]
        y = df_features['sales_quantity']
        
        # 5. 训练模型
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = LightGBMModel()
        model.fit(X_train, y_train)
        
        # 6. 评估
        y_pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 7. 保存模型
        model_dir = Path("models/batch")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{gcode}_{customer[:20].replace('/', '_')}.txt"
        model.save(str(model_path))
        
        print(f"  ✅ 成功 - RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return {
            'gcode': gcode,
            'customer': customer,
            'status': 'success',
            'model_path': str(model_path),
            'record_count': len(df),
            'feature_count': len(feature_cols),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return {
            'gcode': gcode,
            'customer': customer,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """主函数"""
    
    # 定义要训练的组合
    combinations = [
        ('026436', '柳州市工人医院'),
        ('026436', '桂林医学院附属医院'),
        ('026436', '广西壮族自治区人民医院'),
        # 添加更多组合...
    ]
    
    print(f"开始批量训练 - 总计 {len(combinations)} 个组合")
    
    results = []
    for gcode, customer in combinations:
        result = train_single(gcode, customer)
        results.append(result)
    
    # 保存结果报告
    report_df = pd.DataFrame(results)
    report_df.to_csv('batch_training_report.csv', index=False)
    
    # 统计
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n{'='*80}")
    print(f"批量训练完成!")
    print(f"{'='*80}")
    print(f"总计: {len(results)} 个组合")
    print(f"成功: {success_count} 个")
    print(f"跳过: {skipped_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"\n报告已保存: batch_training_report.csv")

if __name__ == "__main__":
    main()
```

### 运行

```bash
python scripts/simple_batch_train.py
```

---

## 方法二：并行批量训练（推荐⭐）

### 适用场景
- 组合数量多（>50个）
- 有多核CPU
- 需要加速训练

### 完整代码

```python
# scripts/parallel_batch_train.py

from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 复用 simple_batch_train.py 的 train_single 函数
from simple_batch_train import train_single
import pandas as pd

def parallel_batch_train(combinations, max_workers=4):
    """并行批量训练"""
    
    print(f"开始并行批量训练")
    print(f"总计: {len(combinations)} 个组合")
    print(f"并行度: {max_workers} 个进程")
    
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(train_single, gcode, customer): (gcode, customer)
            for gcode, customer in combinations
        }
        
        # 收集结果
        for future in as_completed(future_to_task):
            gcode, customer = future_to_task[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                status = result['status']
                print(f"[{completed}/{len(combinations)}] {gcode} - {customer}: {status}")
                
            except Exception as e:
                print(f"[{completed}/{len(combinations)}] {gcode} - {customer}: 异常 - {e}")
                results.append({
                    'gcode': gcode,
                    'customer': customer,
                    'status': 'exception',
                    'error': str(e)
                })
    
    # 保存结果
    report_df = pd.DataFrame(results)
    report_df.to_csv('parallel_batch_report.csv', index=False)
    
    # 统计
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n完成! 成功: {success_count}/{len(results)}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='并行进程数')
    args = parser.parse_args()
    
    # 定义组合
    combinations = [
        ('026436', '柳州市工人医院'),
        ('026436', '桂林医学院附属医院'),
        # ... 更多
    ]
    
    parallel_batch_train(combinations, max_workers=args.workers)
```

### 运行

```bash
# 使用4个进程并行
python scripts/parallel_batch_train.py --workers 4

# 使用8个进程并行
python scripts/parallel_batch_train.py --workers 8
```

---

## 方法三：智能自动选择组合

### 适用场景
- 不知道该训练哪些组合
- 想自动筛选有价值的组合
- 数据量驱动

### 完整代码

```python
# scripts/auto_batch_train.py

from src.data.loader import DataLoader
import pandas as pd

def auto_select_combinations(min_records=500, top_n_products=20, top_n_customers=10):
    """自动选择要训练的药品-客户组合"""
    
    print("自动选择训练组合...")
    
    loader = DataLoader()
    
    # 1. 获取所有药品
    all_gcodes = loader.get_unique_gcodes()
    valid_gcodes = [g for g in all_gcodes if g and str(g).strip() and str(g) != 'None']
    
    print(f"有效药品数量: {len(valid_gcodes)}")
    
    # 2. 统计每个药品的记录数
    product_stats = []
    
    for i, gcode in enumerate(valid_gcodes[:100], 1):  # 只检查前100个
        if i % 10 == 0:
            print(f"  检查药品 {i}/100...")
        
        df = loader.load_sales_data(gcode=gcode, limit=1)
        # 使用 SQL COUNT 查询总记录数
        query = f"""
        SELECT COUNT(*) as count 
        FROM datasense_dlink_erpservice.view_dws_erp_sal_detail_df 
        WHERE gcode = '{gcode}'
        """
        count_df = pd.read_sql(query, loader.db_manager.engine)
        count = count_df['count'].iloc[0]
        
        if count >= min_records:
            product_stats.append({
                'gcode': gcode,
                'total_records': count
            })
    
    # 3. 选择记录数最多的前N个药品
    product_stats_df = pd.DataFrame(product_stats)
    product_stats_df = product_stats_df.sort_values('total_records', ascending=False)
    top_products = product_stats_df.head(top_n_products)
    
    print(f"\n选择的药品 (Top {top_n_products}):")
    print(top_products.to_string(index=False))
    
    # 4. 对每个药品，选择前N个客户
    combinations = []
    
    for _, row in top_products.iterrows():
        gcode = row['gcode']
        print(f"\n分析药品 {gcode} 的客户...")
        
        # 查询该药品的客户统计
        query = f"""
        SELECT 
            cust_name,
            COUNT(*) as record_count,
            SUM(qty) as total_qty
        FROM datasense_dlink_erpservice.view_dws_erp_sal_detail_df
        WHERE gcode = '{gcode}'
        GROUP BY cust_name
        ORDER BY record_count DESC
        LIMIT {top_n_customers}
        """
        
        customer_stats = pd.read_sql(query, loader.db_manager.engine)
        
        for _, cust_row in customer_stats.iterrows():
            customer = cust_row['cust_name']
            if customer and str(customer).strip():
                combinations.append((gcode, customer))
                print(f"  + {customer}: {cust_row['record_count']} 条记录")
    
    print(f"\n总计选择: {len(combinations)} 个组合")
    
    # 5. 保存组合列表
    combo_df = pd.DataFrame(combinations, columns=['gcode', 'customer'])
    combo_df.to_csv('selected_combinations.csv', index=False)
    print("组合列表已保存: selected_combinations.csv")
    
    return combinations

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-records', type=int, default=500)
    parser.add_argument('--top-products', type=int, default=20)
    parser.add_argument('--top-customers', type=int, default=10)
    parser.add_argument('--parallel', type=int, default=4)
    args = parser.parse_args()
    
    # 自动选择组合
    combinations = auto_select_combinations(
        min_records=args.min_records,
        top_n_products=args.top_products,
        top_n_customers=args.top_customers
    )
    
    # 并行训练
    if combinations:
        from parallel_batch_train import parallel_batch_train
        parallel_batch_train(combinations, max_workers=args.parallel)
```

### 运行

```bash
# 自动选择并训练
python scripts/auto_batch_train.py \
    --min-records 500 \
    --top-products 20 \
    --top-customers 10 \
    --parallel 4
```

---

## 方法四：使用配置文件

### 配置文件

```yaml
# config/batch_config.yaml

batch_training:
  # 自动选择配置
  auto_select:
    enabled: true
    min_records: 500
    top_n_products: 20
    top_n_customers: 10
  
  # 或手动指定
  manual_combinations:
    - gcode: "026436"
      customers:
        - "柳州市工人医院"
        - "桂林医学院附属医院"
        - "广西壮族自治区人民医院"
    
    - gcode: "026437"
      customers:
        - "南宁市第一人民医院"
  
  # 训练配置
  training:
    test_size: 0.2
    min_train_samples: 100
    model_type: "lightgbm"
  
  # 超参数优化（可选）
  hyperparameter_tuning:
    enabled: false
    method: "optuna"  # optuna, grid_search, random_search
    n_trials: 50
  
  # 输出配置
  output:
    model_dir: "models/batch"
    report_file: "reports/batch_training_report.csv"
    log_dir: "logs/batch"
  
  # 并行配置
  parallel:
    enabled: true
    max_workers: 4
```

### 使用配置文件的脚本

```python
# scripts/batch_train_from_config.py

import yaml
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['batch_training']

def main(config_path='config/batch_config.yaml'):
    config = load_config(config_path)
    
    # 获取组合
    if config['auto_select']['enabled']:
        from auto_batch_train import auto_select_combinations
        combinations = auto_select_combinations(
            min_records=config['auto_select']['min_records'],
            top_n_products=config['auto_select']['top_n_products'],
            top_n_customers=config['auto_select']['top_n_customers']
        )
    else:
        # 手动指定的组合
        combinations = []
        for item in config['manual_combinations']:
            gcode = item['gcode']
            for customer in item['customers']:
                combinations.append((gcode, customer))
    
    # 并行训练
    if config['parallel']['enabled']:
        from parallel_batch_train import parallel_batch_train
        parallel_batch_train(
            combinations,
            max_workers=config['parallel']['max_workers']
        )
    else:
        from simple_batch_train import train_single
        for gcode, customer in combinations:
            train_single(gcode, customer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/batch_config.yaml')
    args = parser.parse_args()
    
    main(args.config)
```

### 运行

```bash
python scripts/batch_train_from_config.py --config config/batch_config.yaml
```

---

## 批量预测

训练完成后，使用模型进行批量预测：

```python
# scripts/batch_predict.py

from pathlib import Path
import pandas as pd
from src.models.lgb_model import LightGBMModel

def batch_predict(model_dir, forecast_days=30):
    """批量预测"""
    
    model_dir = Path(model_dir)
    model_files = list(model_dir.glob("*.txt"))
    
    print(f"找到 {len(model_files)} 个模型")
    
    predictions = []
    
    for model_file in model_files:
        # 从文件名解析 gcode 和 customer
        # 例如: 026436_柳州市工人医院.txt
        parts = model_file.stem.split('_', 1)
        gcode = parts[0]
        customer = parts[1] if len(parts) > 1 else "unknown"
        
        print(f"预测: {gcode} - {customer}")
        
        # 加载模型
        model = LightGBMModel()
        model.load(str(model_file))
        
        # 准备预测数据
        # ... (加载最新数据，构建特征)
        
        # 预测
        # forecast = model.predict(X_future)
        
        # predictions.append({...})
    
    # 保存预测结果
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv('batch_predictions.csv', index=False)
    
    return predictions
```

---

## 模型管理

### 模型命名规范

```
models/batch/
├── 026436_柳州市工人医院.txt
├── 026436_桂林医学院附属医院.txt
├── 026437_南宁市第一人民医院.txt
└── ...
```

### 模型元数据

```json
{
  "gcode": "026436",
  "customer": "柳州市工人医院",
  "model_path": "models/batch/026436_柳州市工人医院.txt",
  "train_date": "2025-11-05",
  "data_range": ["2020-01-01", "2024-12-31"],
  "metrics": {
    "rmse": 45.23,
    "mae": 32.15,
    "r2": 0.8542
  },
  "feature_count": 47,
  "train_size": 1388
}
```

---

## 最佳实践

### 1. 数据筛选
- 最少记录数：500-1000条
- 数据时间跨度：至少1年
- 剔除异常组合

### 2. 并行策略
- CPU密集型：max_workers = CPU核心数
- I/O密集型：max_workers = CPU核心数 × 2
- 建议：4-8个进程

### 3. 错误处理
- 记录所有失败案例
- 保存中间结果
- 支持断点续训

### 4. 监控和日志
- 每个模型独立日志
- 记录训练时间
- 监控资源使用

### 5. 模型版本管理
- 包含训练日期
- 保存训练参数
- 记录数据版本

---

## 完整工作流

```bash
# 1. 自动选择组合
python scripts/auto_select_combinations.py \
    --min-records 500 \
    --top-products 20 \
    --top-customers 10

# 2. 批量训练
python scripts/parallel_batch_train.py \
    --input selected_combinations.csv \
    --workers 4

# 3. 查看报告
cat parallel_batch_report.csv

# 4. 批量预测
python scripts/batch_predict.py \
    --model-dir models/batch \
    --forecast-days 30
```

---

## 总结

| 方法 | 组合数 | 速度 | 复杂度 | 推荐度 |
|------|--------|------|--------|--------|
| 简单循环 | <50 | 慢 | 低 | ⭐⭐⭐ |
| 并行训练 | 50-500 | 快 | 中 | ⭐⭐⭐⭐⭐ |
| 自动选择 | 任意 | 快 | 中 | ⭐⭐⭐⭐⭐ |
| 配置文件 | 任意 | 快 | 低 | ⭐⭐⭐⭐ |
