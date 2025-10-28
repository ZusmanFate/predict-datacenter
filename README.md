# 药品销量时间序列预测系统

## 📊 项目概述

这是一个工程化的时间序列预测系统，用于预测单个药品在单个医院维度的销量，并支持扩展到多个药品/医院的批量预测。

## 🏗️ 系统架构

```
数据层（ODS → DW） → 特征工程（ETL + Feature Store） → 训练（Model Train） → 
评估（AutoML / Cross Validation） → 部署（Batch or API） → 监控（Drift Detection）
```

## 🎯 核心特性

- ✅ 数据库连接与数据获取
- ✅ 自动化特征工程
- ✅ 批量建模与预测（支持多药品/医院）
- ✅ 模型版本管理（MLflow）
- ✅ REST API 服务（FastAPI）
- ✅ 模型监控与漂移检测
- ✅ 自动调参与优化（Optuna）

## 📦 技术栈

| 模块 | 技术 |
|------|------|
| 数据处理 | pandas, PySpark |
| 特征工程 | pandas, numpy, scikit-learn |
| 模型训练 | LightGBM, XGBoost, CatBoost, Prophet |
| 模型管理 | MLflow |
| 超参数优化 | Optuna |
| API 服务 | FastAPI, uvicorn |
| 数据库 | SQLAlchemy（支持 MySQL, PostgreSQL, SQLite） |
| 任务调度 | APScheduler |
| 监控 | Prometheus, Grafana（可选） |
| 部署 | Docker, Docker Compose |

## 📁 项目结构

```
预测模型工程/
├── config/                 # 配置文件
│   ├── config.yaml        # 主配置文件
│   ├── database.yaml      # 数据库配置
│   └── model_config.yaml  # 模型配置
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── features/         # 特征数据
├── src/                   # 源代码
│   ├── data/             # 数据层
│   │   ├── database.py   # 数据库连接
│   │   ├── loader.py     # 数据加载
│   │   └── processor.py  # 数据预处理
│   ├── features/         # 特征工程
│   │   ├── builder.py    # 特征构建
│   │   └── store.py      # 特征存储
│   ├── models/           # 模型层
│   │   ├── base.py       # 基础模型类
│   │   ├── lgb_model.py  # LightGBM 模型
│   │   ├── prophet_model.py # Prophet 模型
│   │   └── ensemble.py   # 集成模型
│   ├── training/         # 训练模块
│   │   ├── trainer.py    # 训练器
│   │   ├── evaluator.py  # 评估器
│   │   └── optimizer.py  # 超参数优化
│   ├── serving/          # 服务层
│   │   ├── api.py        # FastAPI 服务
│   │   └── predictor.py  # 预测器
│   ├── monitoring/       # 监控模块
│   │   ├── drift.py      # 数据漂移检测
│   │   └── metrics.py    # 指标监控
│   └── utils/            # 工具函数
│       ├── logger.py     # 日志工具
│       └── helpers.py    # 辅助函数
├── notebooks/            # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── tests/                # 单元测试
│   └── test_*.py
├── scripts/              # 脚本
│   ├── train.py         # 训练脚本
│   ├── predict.py       # 预测脚本
│   └── deploy.py        # 部署脚本
├── mlruns/              # MLflow 实验记录
├── logs/                # 日志文件
├── requirements.txt     # Python 依赖
├── Dockerfile           # Docker 配置
├── docker-compose.yml   # Docker Compose 配置
└── README.md           # 项目文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置数据库

编辑 `config/database.yaml`，配置您的数据库连接信息。

### 3. 数据准备

```bash
# 从数据库加载数据
python scripts/load_data.py
```

### 4. 特征工程

```bash
# 构建特征
python scripts/build_features.py
```

### 5. 模型训练

```bash
# 训练模型
python scripts/train.py --model lightgbm --drug_id 001 --hospital_id H001
```

### 6. 启动 API 服务

```bash
# 启动 FastAPI 服务
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

访问 API 文档：http://localhost:8000/docs

### 7. 批量预测

```bash
# 批量预测
python scripts/predict.py --input data/processed/test.csv --output predictions.csv
```

## 📊 开发路线图

### 阶段 1：快速验证（1周）
- [x] 项目结构搭建
- [ ] 数据库连接与数据加载
- [ ] 基础特征工程
- [ ] Prophet 基准模型
- [ ] 验证预测效果

### 阶段 2：工程化原型（2-3周）
- [ ] LightGBM 批量建模
- [ ] MLflow 模型管理
- [ ] 特征存储优化
- [ ] 批量预测管线

### 阶段 3：自动优化（2周）
- [ ] Optuna 超参数优化
- [ ] AutoML 集成
- [ ] 模型集成策略

### 阶段 4：上线部署（1周）
- [ ] FastAPI REST 服务
- [ ] Docker 容器化
- [ ] 任务调度（Airflow/APScheduler）

### 阶段 5：监控优化（持续）
- [ ] 数据漂移检测
- [ ] 模型性能监控
- [ ] 自动再训练机制

## 📝 使用示例

### Python API

```python
from src.models.lgb_model import LightGBMModel
from src.training.trainer import ModelTrainer

# 创建模型
model = LightGBMModel()

# 训练
trainer = ModelTrainer(model)
trainer.train(train_data, valid_data)

# 预测
predictions = model.predict(test_data)
```

### REST API

```bash
# 健康检查
curl http://localhost:8000/health

# 单次预测
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"drug_id": "001", "hospital_id": "H001", "date": "2024-01-01"}'

# 批量预测
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d @batch_input.json
```

## 🔧 配置说明

所有配置文件位于 `config/` 目录下：

- `config.yaml`: 主配置（项目路径、日志等）
- `database.yaml`: 数据库连接配置
- `model_config.yaml`: 模型超参数配置

## 📈 模型评估指标

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

## 🐛 常见问题

1. **数据库连接失败**：检查 `config/database.yaml` 配置是否正确
2. **模型训练慢**：可以调整 `model_config.yaml` 中的参数，或使用更少的数据进行快速验证
3. **内存不足**：考虑使用 PySpark 进行分布式处理

## 📚 参考文档

- [LightGBM 文档](https://lightgbm.readthedocs.io/)
- [Prophet 文档](https://facebook.github.io/prophet/)
- [MLflow 文档](https://mlflow.org/docs/latest/index.html)
- [FastAPI 文档](https://fastapi.tiangolo.com/)

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

## 📄 License

MIT License
