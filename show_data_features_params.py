"""
展示数据库字段、特征工程、模型参数的完整示例
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.models.lgb_model import LightGBMModel
from src.utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

def show_database_schema():
    """展示数据库原始字段"""
    print("\n" + "=" * 100)
    print(" 1️⃣  数据库原始字段 ".center(100, "="))
    print("=" * 100)
    
    loader = DataLoader()
    
    # 加载少量数据查看结构
    print("\n📋 从 StarRocks 加载的原始数据字段:")
    df = loader.load_sales_data(gcode="026436", limit=5)
    
    print(f"\n数据表: datasense_dlink_erpservice.view_dws_erp_sal_detail_df")
    print(f"总列数: {len(df.columns)}")
    print("\n字段列表及数据类型:")
    print("-" * 100)
    
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        sample = df[col].iloc[0] if len(df) > 0 else "N/A"
        print(f"  {i:2d}. {col:30s} | 类型: {str(dtype):10s} | 示例: {sample}")
    
    print("\n📊 数据样本（前3条）:")
    print("-" * 100)
    print(df[['gcode', 'create_dt', 'qty', 'invoice_price', 'cust_name', 'gname']].head(3).to_string(index=False))
    
    print("\n💡 关键字段说明:")
    print("  • gcode: 药品编码（用于筛选特定药品）")
    print("  • create_dt: 销售日期（时间序列的关键字段）")
    print("  • qty: 销售数量（预测目标）")
    print("  • cust_name: 客户名称（用于区分不同医院/客户）")
    print("  • invoice_price: 开票价格")
    print("  • gname: 药品名称")
    
    return df


def show_feature_engineering(df):
    """展示特征工程过程"""
    print("\n" + "=" * 100)
    print(" 2️⃣  特征工程详解 ".center(100, "="))
    print("=" * 100)
    
    # 准备数据
    top_customer = df.groupby('cust_name').size().idxmax()
    df = df[df['cust_name'] == top_customer]
    
    # 重命名和聚合
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
    
    # 数据预处理
    processor = DataProcessor()
    df_proc = processor.create_time_series_dataset(
        df_proc, drug_id="026436", hospital_id=top_customer,
        date_column='date', target_column='sales_quantity'
    )
    df_proc = processor.handle_missing_values(df_proc, method='forward_fill')
    df_proc = processor.handle_outliers(df_proc, 'sales_quantity', method='iqr')
    
    print(f"\n📐 预处理后的数据:")
    print(f"  • 数据量: {len(df_proc)} 条（按天聚合后）")
    print(f"  • 日期范围: {df_proc['date'].min()} 至 {df_proc['date'].max()}")
    print(f"  • 目标变量统计:")
    print(f"    - 最小值: {df_proc['sales_quantity'].min():.0f}")
    print(f"    - 最大值: {df_proc['sales_quantity'].max():.0f}")
    print(f"    - 平均值: {df_proc['sales_quantity'].mean():.2f}")
    print(f"    - 标准差: {df_proc['sales_quantity'].std():.2f}")
    
    # 特征工程
    print("\n🔧 开始特征工程...")
    feature_builder = FeatureBuilder()
    df_features = feature_builder.build_features(
        df_proc, target_column='sales_quantity', date_column='date'
    )
    
    print(f"\n✅ 特征构建完成！")
    print(f"  • 总特征数: {len(df_features.columns)} 列")
    print(f"  • 数据量: {len(df_features)} 条")
    
    # 按类别展示特征
    print("\n📊 特征分类详解:")
    print("-" * 100)
    
    # 1. 日期特征
    date_features = [col for col in df_features.columns if any(x in col for x in ['year', 'month', 'day', 'dayofweek', 'quarter', 'sin', 'cos'])]
    print(f"\n1️⃣  日期特征 ({len(date_features)} 个):")
    for feat in date_features:
        print(f"  • {feat}")
    
    # 2. 滞后特征
    lag_features = [col for col in df_features.columns if 'lag' in col]
    print(f"\n2️⃣  滞后特征 ({len(lag_features)} 个) - 历史销量:")
    for feat in lag_features:
        print(f"  • {feat}")
    
    # 3. 滚动窗口特征
    rolling_features = [col for col in df_features.columns if 'rolling' in col]
    print(f"\n3️⃣  滚动窗口特征 ({len(rolling_features)} 个) - 统计特征:")
    for feat in rolling_features[:10]:  # 只显示前10个
        print(f"  • {feat}")
    if len(rolling_features) > 10:
        print(f"  • ... 还有 {len(rolling_features) - 10} 个")
    
    # 4. 差分特征
    diff_features = [col for col in df_features.columns if 'diff' in col]
    print(f"\n4️⃣  差分特征 ({len(diff_features)} 个) - 变化率:")
    for feat in diff_features:
        print(f"  • {feat}")
    
    # 5. 统计特征
    stat_features = [col for col in df_features.columns if 'cumsum' in col or 'cummean' in col or 'cumstd' in col]
    print(f"\n5️⃣  统计特征 ({len(stat_features)} 个) - 累计统计:")
    for feat in stat_features:
        print(f"  • {feat}")
    
    # 显示特征数据样本
    print("\n📈 特征数据样本（最后3条）:")
    print("-" * 100)
    display_cols = ['date', 'sales_quantity', 'sales_quantity_lag_1', 'sales_quantity_rolling_7_mean', 
                    'sales_quantity_diff_1', 'dayofweek']
    print(df_features[display_cols].tail(3).to_string(index=False))
    
    return df_features


def show_model_params():
    """展示模型参数"""
    print("\n" + "=" * 100)
    print(" 3️⃣  模型参数详解 ".center(100, "="))
    print("=" * 100)
    
    print("\n🤖 LightGBM 模型默认参数:")
    print("-" * 100)
    
    model = LightGBMModel()
    params = model.model_params
    
    print("\n核心参数:")
    for key, value in params.items():
        description = get_param_description(key)
        print(f"  • {key:25s} = {str(value):10s}  # {description}")
    
    print("\n\n📖 参数说明:")
    print("-" * 100)
    print("""
1. n_estimators (100)
   - 含义: 树的数量
   - 影响: 越多越复杂，但可能过拟合
   - 建议: 50-500

2. max_depth (6)
   - 含义: 树的最大深度
   - 影响: 越深模型越复杂
   - 建议: 3-10

3. learning_rate (0.1)
   - 含义: 学习率
   - 影响: 越小需要更多树，但更稳定
   - 建议: 0.01-0.3

4. num_leaves (31)
   - 含义: 叶子节点数量
   - 影响: 控制模型复杂度
   - 建议: 20-100

5. min_child_samples (20)
   - 含义: 叶子节点最小样本数
   - 影响: 防止过拟合
   - 建议: 10-100

6. subsample (0.8)
   - 含义: 每次迭代使用的数据比例
   - 影响: 防止过拟合
   - 建议: 0.6-1.0

7. colsample_bytree (0.8)
   - 含义: 每棵树使用的特征比例
   - 影响: 防止过拟合
   - 建议: 0.6-1.0
    """)
    
    return params


def get_param_description(param_name):
    """获取参数描述"""
    descriptions = {
        'objective': '目标函数（回归任务）',
        'metric': '评估指标',
        'n_estimators': '树的数量',
        'max_depth': '树的最大深度',
        'learning_rate': '学习率',
        'num_leaves': '叶子节点数',
        'min_child_samples': '叶子最小样本数',
        'subsample': '数据采样比例',
        'colsample_bytree': '特征采样比例',
        'random_state': '随机种子',
        'n_jobs': '并行线程数',
        'verbose': '是否打印训练日志'
    }
    return descriptions.get(param_name, '其他参数')


def show_hyperparameter_tuning():
    """展示超参数调优方法"""
    print("\n" + "=" * 100)
    print(" 4️⃣  超参数调优方法 ".center(100, "="))
    print("=" * 100)
    
    print("\n📚 方法一：网格搜索 (Grid Search)")
    print("-" * 100)
    print("""
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'num_leaves': [20, 31, 50]
}

# 网格搜索
model = LightGBMModel()
grid_search = GridSearchCV(
    model.model,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳得分:", -grid_search.best_score_)
    """)
    
    print("\n📚 方法二：随机搜索 (Random Search)")
    print("-" * 100)
    print("""
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 定义参数分布
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'num_leaves': randint(20, 100),
    'min_child_samples': randint(10, 100)
}

# 随机搜索
random_search = RandomizedSearchCV(
    model.model,
    param_distributions,
    n_iter=50,  # 尝试50次随机组合
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
random_search.fit(X_train, y_train)
    """)
    
    print("\n📚 方法三：Optuna 贝叶斯优化 (推荐)")
    print("-" * 100)
    print("""
import optuna
from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer

# 定义搜索空间
search_space = {
    'n_estimators': ('int', 50, 500),
    'max_depth': ('int', 3, 10),
    'learning_rate': ('float', 0.01, 0.3),
    'num_leaves': ('int', 20, 100),
    'min_child_samples': ('int', 10, 100),
    'subsample': ('float', 0.6, 1.0),
    'colsample_bytree': ('float', 0.6, 1.0)
}

# 创建优化器
optimizer = HyperparameterOptimizer(
    model=LightGBMModel(),
    search_space=search_space
)

# 运行优化
best_params = optimizer.optimize(
    X_train, y_train,
    n_trials=100,  # 尝试100次
    cv=5
)

print("最佳参数:", best_params)
    """)
    
    print("\n💡 实用脚本示例:")
    print("-" * 100)
    print("""
# 保存为: scripts/tune_hyperparameters.py

python scripts/tune_hyperparameters.py \\
    --gcode 026436 \\
    --method optuna \\
    --n-trials 100 \\
    --output models/best_params_026436.json
    """)


def show_batch_training():
    """展示批量建模方法"""
    print("\n" + "=" * 100)
    print(" 5️⃣  批量建模方法 ".center(100, "="))
    print("=" * 100)
    
    print("\n📚 方法一：简单循环批量训练")
    print("-" * 100)
    print("""
from src.data.loader import DataLoader
from src.training.trainer import ModelTrainer
from src.models.lgb_model import LightGBMModel

loader = DataLoader()

# 获取所有药品-客户组合
gcodes = ['026436', '026437', '026438']  # 药品列表
customers = ['柳州市工人医院', '桂林医学院附属医院']  # 客户列表

results = []

for gcode in gcodes:
    for customer in customers:
        try:
            print(f"训练: {gcode} - {customer}")
            
            # 加载数据
            df = loader.load_sales_data(gcode=gcode)
            df = df[df['cust_name'] == customer]
            
            if len(df) < 100:
                print(f"  跳过: 数据不足")
                continue
            
            # 训练模型
            model = LightGBMModel()
            trainer = ModelTrainer(model, experiment_name=f"{gcode}_{customer}")
            
            # ... 特征工程和训练 ...
            
            # 保存模型
            model_path = f"models/{gcode}_{customer}.txt"
            model.save(model_path)
            
            results.append({
                'gcode': gcode,
                'customer': customer,
                'model_path': model_path,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  失败: {e}")
            results.append({
                'gcode': gcode,
                'customer': customer,
                'status': 'failed',
                'error': str(e)
            })

# 保存结果
import pandas as pd
pd.DataFrame(results).to_csv('batch_training_results.csv', index=False)
    """)
    
    print("\n📚 方法二：使用配置文件批量训练")
    print("-" * 100)
    print("""
# 创建配置文件: config/batch_config.yaml

batch_training:
  # 从数据库自动筛选
  auto_select:
    enabled: true
    min_records: 500  # 最少记录数
    top_n_products: 20  # 前N个药品
    top_n_customers: 10  # 每个药品的前N个客户
  
  # 或手动指定
  manual_list:
    - gcode: "026436"
      customers: ["柳州市工人医院", "桂林医学院附属医院"]
    - gcode: "026437"
      customers: ["广西壮族自治区人民医院"]
  
  # 训练参数
  training:
    test_size: 0.2
    model_type: "lightgbm"
    use_optuna: true
    n_trials: 50
  
  # 输出设置
  output:
    model_dir: "models/batch"
    log_dir: "logs/batch"
    report_file: "reports/batch_training_report.html"

# 运行批量训练
python scripts/batch_train.py --config config/batch_config.yaml
    """)
    
    print("\n📚 方法三：并行批量训练（推荐）")
    print("-" * 100)
    print("""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def train_single_model(gcode, customer, config):
    '''训练单个模型'''
    try:
        # ... 训练逻辑 ...
        return {'gcode': gcode, 'customer': customer, 'status': 'success'}
    except Exception as e:
        return {'gcode': gcode, 'customer': customer, 'status': 'failed', 'error': str(e)}

# 准备任务列表
tasks = [
    ('026436', '柳州市工人医院'),
    ('026436', '桂林医学院附属医院'),
    ('026437', '广西壮族自治区人民医院'),
    # ... 更多组合
]

# 并行训练
results = []
with ProcessPoolExecutor(max_workers=4) as executor:
    # 提交所有任务
    future_to_task = {
        executor.submit(train_single_model, gcode, customer, config): (gcode, customer)
        for gcode, customer in tasks
    }
    
    # 收集结果
    for future in as_completed(future_to_task):
        gcode, customer = future_to_task[future]
        try:
            result = future.result()
            results.append(result)
            print(f"完成: {gcode} - {customer}")
        except Exception as e:
            print(f"失败: {gcode} - {customer}: {e}")

print(f"\\n总计: {len(results)} 个模型训练完成")
    """)
    
    print("\n💡 实用脚本示例:")
    print("-" * 100)
    print("""
# 1. 自动批量训练（推荐）
python scripts/batch_train.py \\
    --auto \\
    --min-records 500 \\
    --top-products 20 \\
    --top-customers 10 \\
    --parallel 4

# 2. 从文件批量训练
python scripts/batch_train.py \\
    --input config/product_customer_list.csv \\
    --parallel 4 \\
    --output-dir models/batch

# 3. 批量预测
python scripts/batch_predict.py \\
    --model-dir models/batch \\
    --forecast-days 30 \\
    --output results/batch_forecast.csv
    """)


def main():
    """主函数"""
    try:
        # 1. 展示数据库字段
        df = show_database_schema()
        
        # 2. 展示特征工程
        df_features = show_feature_engineering(df)
        
        # 3. 展示模型参数
        show_model_params()
        
        # 4. 展示超参数调优
        show_hyperparameter_tuning()
        
        # 5. 展示批量建模
        show_batch_training()
        
        print("\n" + "=" * 100)
        print(" ✅ 所有内容展示完成！".center(100, "="))
        print("=" * 100)
        
        print("\n💡 下一步:")
        print("  1. 查看本脚本生成的详细说明")
        print("  2. 尝试调整模型参数: 修改 src/models/lgb_model.py")
        print("  3. 运行超参数优化: 创建 scripts/tune_hyperparameters.py")
        print("  4. 运行批量建模: 创建 scripts/batch_train.py")
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
