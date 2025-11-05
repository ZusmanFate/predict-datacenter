"""
简化版完整工作流 - gcode=026436
快速验证和展示完整流程
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.models.lgb_model import LightGBMModel
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

def main():
    """简化版工作流"""
    
    print("\n" + "=" * 100)
    print(" 药品销量预测系统 - 简化完整工作流 gcode=026436 ".center(100, "="))
    print("=" * 100)
    
    try:
        GCODE = "026436"
        START_DATE = "2020-01-01"
        END_DATE = "2024-12-31"
        
        # ========== 步骤1: 数据加载 ==========
        print("\n[步骤 1/7] 数据加载...")
        loader = DataLoader()
        df = loader.load_sales_data(gcode=GCODE)
        df = df[(df['create_dt'] >= START_DATE) & (df['create_dt'] <= END_DATE)]
        print(f"✅ 加载 {len(df)} 条记录，客户数: {df['cust_name'].nunique()}")
        
        # ========== 步骤2: 选择主要客户 ==========
        print("\n[步骤 2/7] 选择主要客户...")
        top_customer = df.groupby('cust_name').size().idxmax()
        df = df[df['cust_name'] == top_customer]
        print(f"✅ 选择客户: {top_customer}, 数据量: {len(df)}")
        
        # ========== 步骤3: 数据预处理 ==========
        print("\n[步骤 3/7] 数据预处理...")
        processor = DataProcessor()
        
        # 重命名并聚合
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
        
        print(f"   聚合后: {len(df_proc)} 条（每天一条）")
        
        # 创建时间序列
        df_proc = processor.create_time_series_dataset(
            df_proc, drug_id=GCODE, hospital_id=top_customer,
            date_column='date', target_column='sales_quantity'
        )
        df_proc = processor.handle_missing_values(df_proc, method='forward_fill')
        df_proc = processor.handle_outliers(df_proc, 'sales_quantity', method='iqr')
        print(f"✅ 预处理完成: {len(df_proc)} 条")
        
        # ========== 步骤4: 特征工程 ==========
        print("\n[步骤 4/7] 特征工程...")
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(
            df_proc, target_column='sales_quantity', date_column='date'
        )
        print(f"✅ 特征构建完成: {len(df_features)} 条, {len(df_features.columns)} 列")
        
        # **关键：正确过滤特征列**
        print("\n   特征列过滤...")
        print(f"   全部列: {list(df_features.columns)[:10]}...")
        
        # 排除非特征列，只保留数值列
        exclude_cols = {'sales_quantity', 'date', 'drug_id', 'hospital_id'}
        numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols 
                       and df_features[col].dtype.name in numeric_types]
        
        print(f"   特征列数量: {len(feature_cols)}")
        print(f"   前10个特征: {feature_cols[:10]}")
        
        # 验证特征列
        X = df_features[feature_cols]
        y = df_features['sales_quantity']
        print(f"   X shape: {X.shape}, y shape: {y.shape}")
        print(f"   X dtypes: {X.dtypes.value_counts().to_dict()}")
        
        # ========== 步骤5: 训练模型 ==========
        print("\n[步骤 5/7] 训练模型...")
        
        # 手动划分数据
        split_idx = int(len(df_features) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"   训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        model = LightGBMModel()
        model.fit(X_train, y_train)
        print(f"✅ 模型训练完成")
        
        # ========== 步骤6: 模型评估 ==========
        print("\n[步骤 6/7] 模型评估...")
        y_pred = model.predict(X_test)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test.values, y_pred, return_details=True)
        
        print(f"✅ 模型性能:")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   MAE:  {metrics['mae']:.2f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"   R²:   {metrics['r2']:.4f}")
        
        # ========== 步骤7: 特征重要性 ==========
        print("\n[步骤 7/7] 特征重要性...")
        importance_df = model.get_feature_importance()
        print(f"✅ Top 10 特征:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"   {idx+1:2d}. {row['feature']:30s} {row['importance']:>8.0f}")
        
        # ========== 保存模型 ==========
        print("\n[保存模型]...")
        model_path = Path("models") / f"demo_{GCODE}_{top_customer[:20].replace('/', '_')}.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"✅ 模型已保存: {model_path}")
        
        # ========== 完成 ==========
        print("\n" + "=" * 100)
        print(" ✅ 完整工作流演示成功！".center(100, "="))
        print("=" * 100)
        
        print(f"\n📊 总结:")
        print(f"   药品: {GCODE} (低钙腹膜透析液)")
        print(f"   客户: {top_customer}")
        print(f"   数据: {len(df_features)} 条")
        print(f"   特征: {len(feature_cols)} 个")
        print(f"   性能: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")
        
        print(f"\n💡 您已了解完整工作流程！")
        
    except Exception as e:
        logger.error(f"工作流失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
