# 📊 项目总结 - 药品销量时间序列预测系统

## 🎯 项目概述

这是一个**从0到1构建的工程化时间序列预测系统**，用于预测药品在医院维度的销量，支持批量建模和自动化部署。

### 核心特性

✅ **完整的数据管线**：数据库连接 → 数据加载 → 预处理 → 特征工程  
✅ **多模型支持**：LightGBM（主力）、Prophet（基线）、XGBoost（可扩展）  
✅ **工程化训练**：MLflow 版本管理、超参数优化（Optuna）、交叉验证  
✅ **批量处理**：支持多药品/医院组合的并行训练  
✅ **REST API 服务**：FastAPI + Swagger 文档  
✅ **监控与漂移检测**：数据漂移检测、性能监控  
✅ **容器化部署**：Docker + Docker Compose

---

## 📁 项目结构一览

```
预测模型工程/
├── config/                    # 📝 配置文件
│   ├── config.yaml           # 主配置
│   ├── database.yaml         # 数据库配置
│   └── model_config.yaml     # 模型参数配置
│
├── src/                       # 💻 源代码
│   ├── data/                 # 数据层
│   │   ├── database.py       # 数据库管理（支持 MySQL/PostgreSQL/SQLite）
│   │   ├── loader.py         # 数据加载器
│   │   └── processor.py      # 数据预处理
│   │
│   ├── features/             # 特征工程
│   │   └── builder.py        # 特征构建器（滞后、滚动、日期特征等）
│   │
│   ├── models/               # 模型层
│   │   ├── base.py          # 基础模型类
│   │   ├── lgb_model.py     # LightGBM 模型 ⭐
│   │   └── prophet_model.py # Prophet 模型
│   │
│   ├── training/             # 训练模块
│   │   ├── trainer.py       # 训练器（支持 MLflow）
│   │   ├── evaluator.py     # 评估器（指标 + 可视化）
│   │   └── optimizer.py     # 超参数优化（Optuna）
│   │
│   ├── serving/              # 服务层
│   │   └── api.py           # FastAPI REST 服务 🌐
│   │
│   ├── monitoring/           # 监控模块
│   │   └── drift.py         # 数据漂移检测
│   │
│   └── utils/                # 工具函数
│       ├── logger.py         # 日志管理
│       └── helpers.py        # 辅助函数
│
├── scripts/                   # 🔧 脚本
│   ├── generate_sample_data.py  # 生成示例数据 ⚡
│   ├── train.py                 # 单模型训练
│   ├── predict.py               # 模型预测
│   └── batch_train.py           # 批量训练
│
├── examples/                  # 📚 示例
│   └── complete_example.py   # 完整端到端示例
│
├── notebooks/                 # 📓 Jupyter Notebooks（待添加）
│
├── data/                      # 💾 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── features/             # 特征数据
│
├── models/                    # 🤖 模型存储
├── mlruns/                    # 📊 MLflow 实验记录
├── logs/                      # 📋 日志文件
│
├── Dockerfile                 # 🐳 Docker 配置
├── docker-compose.yml         # Docker Compose 配置
├── requirements.txt           # Python 依赖
├── README.md                  # 完整文档
├── QUICKSTART.md              # 快速开始指南 ⚡
├── .gitignore                # Git 忽略文件
├── .env.example              # 环境变量示例
└── start.bat                 # Windows 启动脚本
```

---

## 🚀 快速开始（5分钟上手）

### 1️⃣ 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 生成示例数据

```bash
python scripts/generate_sample_data.py
```

这将创建：
- ✅ 10个药品 × 5个医院 = 50个组合
- ✅ 2022-2024年的时间序列数据
- ✅ SQLite 数据库（`data/sales.db`）

### 3️⃣ 训练第一个模型

```bash
python scripts/train.py --drug_id D001 --hospital_id H001
```

### 4️⃣ 启动 API 服务

```bash
# 方式1：直接启动
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# 方式2：使用启动脚本（Windows）
start.bat

# 方式3：使用 Docker
docker-compose up -d
```

访问 API 文档：http://localhost:8000/docs

---

## 🎓 核心功能使用指南

### 📊 数据加载

```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_sales_data(drug_id='D001', hospital_id='H001')
```

### 🔧 特征工程

```python
from src.features.builder import FeatureBuilder

builder = FeatureBuilder()
df_features = builder.build_features(df, target_column='sales_quantity')
```

自动生成的特征：
- 滞后特征（lag 1-30天）
- 滚动窗口统计（7/14/30/90天）
- 日期特征（年/月/日/周/季度）
- 周期性编码（sin/cos）
- 差分特征

### 🤖 模型训练

```python
from src.models.lgb_model import LightGBMModel
from src.training.trainer import ModelTrainer

model = LightGBMModel()
trainer = ModelTrainer(model)
trained_model, metrics = trainer.train_on_full_data(df_features)
```

### 📈 模型评估

```python
from src.training.evaluator import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_true, y_pred)
evaluator.plot_predictions(y_true, y_pred, save_path='results.png')
```

### 🔍 超参数优化

```python
from src.training.optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(n_trials=50)
best_params = optimizer.optimize_lightgbm(X_train, y_train, X_valid, y_valid)
best_model = optimizer.get_best_model(X_train, y_train)
```

### 🚨 数据漂移检测

```python
from src.monitoring.drift import DriftDetector

detector = DriftDetector(threshold=0.05)
detector.fit_baseline(baseline_df, columns=feature_cols)
drift_results = detector.detect_drift(baseline_df, current_df)
```

### 🌐 API 调用

```bash
# 健康检查
curl http://localhost:8000/health

# 预测
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_id": "D001",
    "hospital_id": "H001",
    "model_path": "models/lightgbm_D001_H001.txt"
  }'
```

---

## 📊 技术栈

| 类别 | 技术 |
|------|------|
| 数据处理 | pandas, numpy, scikit-learn |
| 时间序列模型 | LightGBM ⭐, Prophet, XGBoost |
| 模型管理 | MLflow |
| 超参数优化 | Optuna |
| API 服务 | FastAPI, Uvicorn |
| 数据库 | SQLAlchemy（支持 MySQL/PostgreSQL/SQLite） |
| 监控 | 数据漂移检测、性能监控 |
| 部署 | Docker, Docker Compose |
| 可视化 | Matplotlib, Seaborn, Plotly |

---

## 🎯 已实现的功能

### ✅ 阶段 1：快速验证（已完成）
- [x] 项目结构设计
- [x] 数据库连接（支持多种数据库）
- [x] 数据加载与预处理
- [x] 基础特征工程
- [x] LightGBM 模型实现
- [x] Prophet 基准模型

### ✅ 阶段 2：工程化原型（已完成）
- [x] 批量建模支持
- [x] MLflow 实验跟踪
- [x] 模型评估器（多种指标+可视化）
- [x] 批量预测管线
- [x] 特征重要性分析

### ✅ 阶段 3：自动优化（已完成）
- [x] Optuna 超参数优化
- [x] 交叉验证
- [x] 模型集成准备

### ✅ 阶段 4：上线部署（已完成）
- [x] FastAPI REST 服务
- [x] Docker 容器化
- [x] API 文档（Swagger）
- [x] 批量训练脚本

### ✅ 阶段 5：监控优化（已完成）
- [x] 数据漂移检测
- [x] 模型性能监控
- [x] 评估报告生成

---

## 🔮 下一步建议

### 短期优化（1-2周）
1. **添加更多模型**：XGBoost、CatBoost、SARIMA
2. **AutoML 集成**：AutoGluon、PyCaret
3. **特征存储**：Feast 集成
4. **任务调度**：Airflow/APScheduler 定时训练

### 中期扩展（1个月）
5. **深度学习模型**：LSTM、Transformer、TFT
6. **在线学习**：增量更新模型
7. **A/B 测试**：模型对比与选择
8. **前端界面**：可视化仪表板

### 长期规划（持续）
9. **分布式训练**：PySpark 集成
10. **模型解释性**：SHAP、LIME
11. **生产监控**：Prometheus + Grafana
12. **自动化再训练**：性能下降自动触发

---

## 📚 使用场景

### 🎯 场景 1：单药品预测
```bash
python scripts/train.py --drug_id D001 --hospital_id H001
python scripts/predict.py --model_path models/xxx.txt --drug_id D001 --hospital_id H001
```

### 🎯 场景 2：批量训练
```bash
python scripts/batch_train.py --max_workers 4
```

### 🎯 场景 3：超参数优化
参考 `examples/complete_example.py` 中的优化示例

### 🎯 场景 4：API 集成
集成到现有系统，通过 REST API 调用预测服务

---

## 🛠️ 常见问题解决

### Q1: 数据库连接失败
**解决**: 检查 `config/database.yaml` 配置，确保数据库服务已启动

### Q2: 依赖安装失败
**解决**: 使用国内镜像源
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: MLflow UI 无法访问
**解决**:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Q4: 内存不足
**解决**: 减少特征数量或使用采样，修改 `config/config.yaml`

---

## 📞 支持与贡献

- 📖 完整文档：`README.md`
- ⚡ 快速开始：`QUICKSTART.md`
- 💡 示例代码：`examples/complete_example.py`
- 🌐 API 文档：http://localhost:8000/docs

---

## 🎉 总结

您现在拥有了一个**完整的、工程化的、可扩展的**时间序列预测系统！

### 核心优势
✅ **模块化设计**：易于扩展和维护  
✅ **工程化实践**：MLflow、Docker、API 服务  
✅ **生产就绪**：批量处理、监控、自动化  
✅ **文档完善**：代码注释、使用指南、示例  

### 立即开始
```bash
# 1. 生成数据
python scripts/generate_sample_data.py

# 2. 运行示例
python examples/complete_example.py

# 3. 启动服务
start.bat  # 或 uvicorn src.serving.api:app --reload
```

**祝您使用愉快！** 🚀
