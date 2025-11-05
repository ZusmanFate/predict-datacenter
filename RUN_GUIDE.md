# 🚀 完整运行指南 - 从零开始

本指南将帮助您**一步步运行和熟悉系统**，包括 Impala 数据库连接、Feast 特征存储和 Airflow 定时任务。

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [测试数据库连接](#2-测试数据库连接)
3. [运行示例代码](#3-运行示例代码)
4. [使用 Feast 特征存储](#4-使用-feast-特征存储)
5. [配置 Airflow 定时任务](#5-配置-airflow-定时任务)
6. [训练和预测](#6-训练和预测)
7. [API 服务使用](#7-api-服务使用)
8. [常见问题](#8-常见问题)

---

## 1️⃣ 环境准备

### 步骤 1.1: 创建虚拟环境

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate

# 验证激活
python --version
```

### 步骤 1.2: 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 如果安装较慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**注意**：如果 `impyla` 或 `apache-airflow` 安装失败，可以先跳过：

```bash
# 最小化安装（不包含 Airflow）
pip install numpy pandas scikit-learn lightgbm sqlalchemy pymysql impyla fastapi uvicorn mlflow pyyaml loguru tqdm matplotlib seaborn
```

---

## 2️⃣ 测试数据库连接

### 步骤 2.1: 验证配置

检查 `config/database.yaml` 的 Impala 配置：

```yaml
# Impala 配置
impala:
  host: "172.17.224.214"
  port: 21050
  database: "Impala"
  username: ""
  password: ""
  auth_mechanism: "NOSASL"
  echo: false
```

### 步骤 2.2: 测试连接

创建测试脚本 `test_connection.py`：

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.database import get_db_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_connection():
    """测试数据库连接"""
    try:
        logger.info("=" * 60)
        logger.info("测试 Impala 数据库连接...")
        logger.info("=" * 60)
        
        # 获取数据库管理器
        db_manager = get_db_manager()
        
        # 测试简单查询
        test_query = "SELECT 1 as test"
        result = db_manager.execute_query(test_query)
        
        logger.info(f"✓ 连接成功！查询结果: {result}")
        
        # 测试销量表
        sales_table = db_manager.config['tables']['sales']['name']
        count_query = f"SELECT COUNT(*) as cnt FROM {sales_table} LIMIT 1"
        
        try:
            result = db_manager.execute_query(count_query)
            logger.info(f"✓ 销量表访问成功: {sales_table}")
        except Exception as e:
            logger.warning(f"⚠ 销量表查询失败: {e}")
        
        logger.info("=" * 60)
        logger.info("✅ 数据库连接测试完成！")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 连接失败: {e}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
```

运行测试：

```bash
python test_connection.py
```

**预期输出**：
```
[INFO] 测试 Impala 数据库连接...
[INFO] ✓ 连接成功！
[INFO] ✓ 销量表访问成功: dwd.dwd_erp_mst_biz_all_df
[INFO] ✅ 数据库连接测试完成！
```

---

## 3️⃣ 运行示例代码

### 步骤 3.1: 加载少量数据测试

创建 `test_data_loading.py`：

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_data_loading():
    """测试数据加载"""
    try:
        logger.info("=" * 60)
        logger.info("测试数据加载...")
        logger.info("=" * 60)
        
        loader = DataLoader()
        
        # 获取唯一的药品和医院列表
        logger.info("1. 获取药品和医院列表...")
        gcodes = loader.get_unique_gcodes()
        logger.info(f"   ✓ 找到 {len(gcodes)} 个唯一药品")
        logger.info(f"   前5个药品: {gcodes[:5]}")
        
        cust_names = loader.get_unique_hospitals()
        logger.info(f"   ✓ 找到 {len(cust_names)} 个唯一客户")
        logger.info(f"   前5个客户: {cust_names[:5]}")
        
        # 加载单个药品-医院的数据
        if gcodes and cust_names:
            logger.info("\n2. 加载示例数据...")
            gcode = gcodes[0]
            cust_name = cust_names[0]
            
            logger.info(f"   药品: {gcode}")
            logger.info(f"   客户: {cust_name}")
            
            df = loader.load_sales_data(
                gcode=gcode,
                cust_name=cust_name,
                limit=100  # 只加载100条测试
            )
            
            logger.info(f"   ✓ 成功加载 {len(df)} 条数据")
            logger.info(f"   数据列: {df.columns.tolist()}")
            
            if len(df) > 0:
                logger.info(f"   日期范围: {df['create_dt'].min()} 到 {df['create_dt'].max()}")
                logger.info(f"   平均销量: {df['qty'].mean():.2f}")
            
            logger.info("\n数据样本:")
            print(df.head())
        
        logger.info("\n=" * 60)
        logger.info("✅ 数据加载测试完成！")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
```

运行测试：

```bash
python test_data_loading.py
```

### 步骤 3.2: 运行完整示例（需要调整）

由于您的数据结构与原始示例不同，需要创建适配的示例：

创建 `examples/impala_example.py`：

```python
"""
Impala 数据库完整示例
演示从 Impala 加载数据到模型训练的完整流程
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.models.lgb_model import LightGBMModel
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """完整示例"""
    
    print("=" * 80)
    print("药品销量预测系统 - Impala 数据库示例")
    print("=" * 80)
    
    try:
        # ==================== 步骤 1: 选择数据 ====================
        print("\n[步骤 1/6] 选择药品和客户...")
        loader = DataLoader()
        
        # 获取药品和客户列表
        gcodes = loader.get_unique_gcodes()
        cust_names = loader.get_unique_hospitals()
        
        if not gcodes or not cust_names:
            logger.error("未找到药品或客户数据")
            return
        
        # 选择第一个药品和客户作为示例
        GCODE = gcodes[0]
        CUST_NAME = cust_names[0]
        
        print(f"✓ 选择药品: {GCODE}")
        print(f"✓ 选择客户: {CUST_NAME}")
        
        # ==================== 步骤 2: 加载数据 ====================
        print("\n[步骤 2/6] 加载销量数据...")
        
        df = loader.load_sales_data(
            gcode=GCODE,
            cust_name=CUST_NAME,
            limit=1000  # 限制数据量用于测试
        )
        
        if len(df) < 50:
            logger.error(f"数据量不足: {len(df)} 条")
            return
        
        print(f"✓ 加载了 {len(df)} 条销量记录")
        print(f"  日期范围: {df['create_dt'].min()} 到 {df['create_dt'].max()}")
        print(f"  平均销量: {df['qty'].mean():.2f}")
        
        # ==================== 步骤 3: 数据预处理 ====================
        print("\n[步骤 3/6] 数据预处理...")
        processor = DataProcessor()
        
        # 重命名列以适配预处理器
        df_processed = df.rename(columns={
            'create_dt': 'date',
            'qty': 'sales_quantity',
            'gcode': 'drug_id',
            'cust_name': 'hospital_id'
        })
        
        # 创建时间序列数据集
        df_processed = processor.create_time_series_dataset(
            df_processed,
            drug_id=GCODE,
            hospital_id=CUST_NAME,
            date_column='date',
            target_column='sales_quantity'
        )
        
        # 处理缺失值和异常值
        df_processed = processor.handle_missing_values(df_processed, method='forward_fill')
        df_processed = processor.handle_outliers(df_processed, 'sales_quantity', method='iqr')
        
        print(f"✓ 预处理完成，数据量: {len(df_processed)}")
        
        # ==================== 步骤 4: 特征工程 ====================
        print("\n[步骤 4/6] 特征工程...")
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(
            df_processed,
            target_column='sales_quantity',
            date_column='date'
        )
        
        print(f"✓ 构建了 {len(df_features.columns)} 个特征")
        print(f"  特征数量: {len(df_features)}")
        
        # ==================== 步骤 5: 训练模型 ====================
        print("\n[步骤 5/6] 训练模型...")
        
        model = LightGBMModel()
        trainer = ModelTrainer(model, experiment_name="impala_example")
        
        trained_model, test_metrics = trainer.train_on_full_data(
            df_features,
            target_column='sales_quantity',
            test_size=0.2,
            log_mlflow=False
        )
        
        print("✓ 模型训练完成")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.4f}%")
        print(f"  R²: {test_metrics['r2']:.4f}")
        
        # ==================== 步骤 6: 特征重要性 ====================
        print("\n[步骤 6/6] 特征重要性分析...")
        
        importance_df = trained_model.get_feature_importance()
        
        print("✓ Top 10 重要特征:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.2f}")
        
        # ==================== 保存模型 ====================
        model_path = f"models/impala_example_{GCODE}_{CUST_NAME}.txt"
        trained_model.save(model_path)
        print(f"\n✓ 模型已保存到: {model_path}")
        
        # ==================== 完成 ====================
        print("\n" + "=" * 80)
        print("✅ 示例运行成功完成！")
        print("=" * 80)
        
        print("\n下一步建议:")
        print("  1. 尝试其他药品和客户组合")
        print("  2. 调整特征工程参数: config/config.yaml")
        print("  3. 启动 API 服务: uvicorn src.serving.api:app --reload")
        print("  4. 配置 Feast 特征存储")
        print("  5. 设置 Airflow 定时任务")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

运行示例：

```bash
python examples/impala_example.py
```

---

## 4️⃣ 使用 Feast 特征存储

### 步骤 4.1: 初始化 Feast 仓库

```bash
cd feature_store

# 初始化 Feast（如果还没有）
feast init

# 应用特征定义
feast apply
```

### 步骤 4.2: 准备特征数据

创建 `scripts/prepare_feast_features.py`：

```python
"""
准备特征数据并导入 Feast
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.features.store import FeatureStore, prepare_features_for_feast
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("准备 Feast 特征数据...")
    
    # 加载数据
    loader = DataLoader()
    gcodes = loader.get_unique_gcodes()[:5]  # 前5个药品
    
    # 构建特征
    feature_builder = FeatureBuilder()
    
    all_features = []
    for gcode in gcodes:
        df = loader.load_sales_data(gcode=gcode, limit=1000)
        df_features = feature_builder.build_features(df)
        all_features.append(df_features)
    
    # 合并所有特征
    import pandas as pd
    df_all = pd.concat(all_features, ignore_index=True)
    
    # 准备 Feast 格式
    df_feast = prepare_features_for_feast(df_all)
    
    # 保存到 Parquet
    output_path = "data/features/sales_features.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_feast.to_parquet(output_path)
    
    logger.info(f"✓ 特征数据已保存: {output_path}")
    
    # 物化到 Feast
    store = FeatureStore()
    store.materialize_features(
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    logger.info("✓ 特征已物化到 Feast")

if __name__ == "__main__":
    main()
```

### 步骤 4.3: 从 Feast 获取特征

```python
from src.features.store import FeatureStore

store = FeatureStore()

# 在线特征（用于实时预测）
entity_rows = [
    {"gcode": "D001", "cust_name": "Hospital_A"}
]
features = [
    "sales_features:sales_quantity_lag_1",
    "sales_features:sales_quantity_rolling_7_mean"
]
online_features = store.get_online_features(entity_rows, features)
```

---

## 5️⃣ 配置 Airflow 定时任务

### 步骤 5.1: 初始化 Airflow

```bash
# 设置 Airflow Home
export AIRFLOW_HOME=~/airflow  # Linux/Mac
$env:AIRFLOW_HOME = "$HOME\airflow"  # Windows PowerShell

# 初始化数据库
airflow db init

# 创建管理员用户
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 步骤 5.2: 配置 DAG 路径

编辑 `$AIRFLOW_HOME/airflow.cfg`：

```ini
[core]
dags_folder = D:\预测模型工程\airflow\dags
```

或者创建符号链接：

```bash
# Windows (以管理员运行)
mklink /D %AIRFLOW_HOME%\dags D:\预测模型工程\airflow\dags
```

### 步骤 5.3: 启动 Airflow

```bash
# 启动 Web 服务器
airflow webserver --port 8080

# 新开终端，启动调度器
airflow scheduler
```

访问 Airflow UI：http://localhost:8080

### 步骤 5.4: 启用 DAG

在 Airflow UI 中：
1. 找到 `drug_sales_forecast_daily` DAG
2. 点击开关启用
3. 点击 "Trigger DAG" 手动触发

---

## 6️⃣ 训练和预测

### 单模型训练

```bash
# 使用新的参数名
python scripts/train.py --gcode D001 --cust_name "Hospital_A"

# 或使用旧的参数名（向后兼容）
python scripts/train.py --drug_id D001 --hospital_id H001
```

### 批量训练

```bash
python scripts/batch_train.py --max_workers 4
```

### 预测

```bash
python scripts/predict.py \
  --model_path models/lightgbm_D001_H001.txt \
  --gcode D001 \
  --cust_name "Hospital_A" \
  --output predictions.csv
```

---

## 7️⃣ API 服务使用

### 启动 API

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

### API 调用示例

```bash
# 健康检查
curl http://localhost:8000/health

# 获取药品列表
curl http://localhost:8000/drugs

# 预测
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_id": "D001",
    "hospital_id": "H001",
    "model_path": "models/lightgbm_D001_H001.txt"
  }'
```

访问 API 文档：http://localhost:8000/docs

---

## 8️⃣ 常见问题

### Q1: Impala 连接超时

**解决方案**：
```python
# 检查网络连接
ping 172.17.224.214

# 检查防火墙
# 确保端口 21050 开放
```

### Q2: 特征工程后数据为空

**原因**：滞后特征导致前面的数据被删除

**解决方案**：
```python
# 调整滞后期配置 config/config.yaml
features:
  lag_features: [1, 7, 30]  # 减少滞后期
```

### Q3: 内存不足

**解决方案**：
```python
# 限制数据量
df = loader.load_sales_data(gcode=gcode, limit=5000)

# 或减少特征数量
```

### Q4: Airflow DAG 不显示

**检查步骤**：
1. 确认 DAG 文件在正确路径
2. 检查 Python 语法错误：`python airflow/dags/drug_sales_forecast_dag.py`
3. 查看 Airflow 日志

---

## 🎉 恭喜！

您现在已经：
- ✅ 连接到 Impala 数据库
- ✅ 运行了完整的预测流程
- ✅ 了解了 Feast 特征存储
- ✅ 配置了 Airflow 定时任务
- ✅ 启动了 API 服务

### 📚 下一步学习

1. **优化特征**：根据业务知识调整特征工程
2. **模型调优**：使用 Optuna 进行超参数优化
3. **监控告警**：配置数据漂移检测
4. **扩展模型**：添加 XGBoost、Prophet 等模型
5. **生产部署**：Docker 容器化部署

### 📞 需要帮助？

- 查看日志：`logs/app.log`
- API 文档：http://localhost:8000/docs
- MLflow UI：`mlflow ui`

祝您使用愉快！🚀
