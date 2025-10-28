# 🚀 快速开始指南

本指南将帮助您在 10 分钟内快速上手药品销量预测系统。

## 📋 前置要求

- Python 3.8+
- pip
- （可选）Docker

## 🛠️ 安装步骤

### 方法 1：本地安装（推荐用于开发）

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 方法 2：使用 Docker

```bash
# 1. 构建镜像
docker-compose build

# 2. 启动服务
docker-compose up -d
```

## 📊 第一步：生成示例数据

```bash
python scripts/generate_sample_data.py
```

这将创建：
- 10 个药品 × 5 个医院 = 50 个组合
- 2022-01-01 到 2024-12-31 的销量数据
- SQLite 数据库文件：`data/sales.db`

## 🎯 第二步：训练您的第一个模型

```bash
python scripts/train.py --drug_id D001 --hospital_id H001 --model lightgbm
```

训练完成后，您将看到：
- 模型文件保存在 `models/` 目录
- MLflow 实验记录在 `mlruns/` 目录
- 特征重要性文件

### 查看训练指标

```bash
# 启动 MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

然后在浏览器中访问：http://localhost:5000

## 🔮 第三步：进行预测

```bash
python scripts/predict.py \
  --model_path models/lightgbm_D001_H001_YYYYMMDD_HHMMSS.txt \
  --drug_id D001 \
  --hospital_id H001 \
  --output predictions.csv
```

## 🌐 第四步：启动 API 服务

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

或使用 Docker：

```bash
docker-compose up -d api
```

### 访问 API 文档

在浏览器中打开：http://localhost:8000/docs

### API 使用示例

#### 健康检查

```bash
curl http://localhost:8000/health
```

#### 单次预测

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_id": "D001",
    "hospital_id": "H001",
    "model_path": "models/lightgbm_D001_H001_YYYYMMDD_HHMMSS.txt",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'
```

#### 获取药品列表

```bash
curl http://localhost:8000/drugs
```

#### 获取医院列表

```bash
curl http://localhost:8000/hospitals
```

## 📈 第五步：超参数优化（可选）

```python
from src.training.optimizer import HyperparameterOptimizer
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder

# 加载数据
loader = DataLoader()
df = loader.load_sales_data(drug_id='D001', hospital_id='H001')

# 特征工程
feature_builder = FeatureBuilder()
df_features = feature_builder.build_features(df)

# 划分数据
from sklearn.model_selection import train_test_split
feature_cols = [col for col in df_features.columns if col not in ['sales_quantity', 'date']]
X = df_features[feature_cols]
y = df_features['sales_quantity']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)

# 优化
optimizer = HyperparameterOptimizer(n_trials=50)
best_params = optimizer.optimize_lightgbm(X_train, y_train, X_valid, y_valid)
```

## 📊 使用 Jupyter Notebook

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🔧 常见问题

### Q1: 数据库连接错误

**问题**: `sqlalchemy.exc.OperationalError`

**解决**: 
1. 检查 `config/database.yaml` 配置
2. 确保已运行 `generate_sample_data.py` 创建数据库
3. 如果使用 MySQL/PostgreSQL，确保数据库服务已启动

### Q2: 模型训练很慢

**解决**: 
1. 减少数据量：使用 `--start_date` 和 `--end_date` 参数
2. 调整模型参数：在 `config/model_config.yaml` 中减少 `num_boost_round`
3. 使用更少的特征：修改 `config/config.yaml` 中的特征配置

### Q3: API 启动失败

**解决**: 
1. 检查端口是否被占用：`netstat -ano | findstr :8000`
2. 更换端口：`uvicorn src.serving.api:app --port 8001`
3. 检查日志：查看 `logs/app.log`

### Q4: 内存不足

**解决**: 
1. 批量处理：一次处理少量药品-医院组合
2. 使用采样：在 `DataLoader` 中添加 `limit` 参数
3. 优化特征：减少滞后期和滚动窗口数量

## 📚 下一步

1. **探索数据**: 使用 Jupyter Notebook 进行数据探索
2. **调整特征**: 修改 `config/config.yaml` 中的特征配置
3. **尝试不同模型**: Prophet、XGBoost、集成模型
4. **批量训练**: 为多个药品-医院组合训练模型
5. **部署监控**: 设置数据漂移检测和自动再训练

## 🎓 进阶教程

### 批量训练多个模型

```python
from src.data.loader import DataLoader
from src.training.trainer import ModelTrainer
from src.models.lgb_model import LightGBMModel

loader = DataLoader()
drug_ids = loader.get_unique_drugs()
hospital_ids = loader.get_unique_hospitals()

for drug_id in drug_ids[:5]:  # 前5个药品
    for hospital_id in hospital_ids[:3]:  # 前3个医院
        try:
            df = loader.load_sales_data(drug_id=drug_id, hospital_id=hospital_id)
            # ... 特征工程和训练
            print(f"✅ 完成: {drug_id} - {hospital_id}")
        except Exception as e:
            print(f"❌ 失败: {drug_id} - {hospital_id}: {e}")
```

### 数据漂移监控

```python
from src.monitoring.drift import DriftDetector

detector = DriftDetector(threshold=0.05)

# 拟合基线
detector.fit_baseline(baseline_df, columns=feature_cols)

# 检测漂移
drift_results = detector.detect_drift(baseline_df, current_df, method='ks')

# 生成报告
detector.generate_drift_report(drift_results, 'reports/drift_report.txt')
```

## 💡 最佳实践

1. **版本控制**: 使用 MLflow 跟踪所有实验
2. **定期评估**: 每周检查模型性能
3. **数据质量**: 定期检查数据完整性和异常值
4. **文档记录**: 记录模型配置和业务决策
5. **监控告警**: 设置性能下降自动告警

## 📞 获取帮助

- 查看完整文档：`README.md`
- 查看 API 文档：http://localhost:8000/docs
- 查看配置说明：`config/` 目录下的 YAML 文件

---

**祝您使用愉快！** 🎉
