"""
完整的端到端示例
演示从数据加载到模型预测的完整流程
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
    print("药品销量预测系统 - 完整示例")
    print("=" * 80)
    
    # 配置
    DRUG_ID = 'D001'
    HOSPITAL_ID = 'H001'
    
    try:
        # ==================== 步骤 1: 加载数据 ====================
        print("\n[步骤 1/6] 加载数据...")
        loader = DataLoader()
        df = loader.load_sales_data(
            drug_id=DRUG_ID,
            hospital_id=HOSPITAL_ID
        )
        print(f"✓ 加载了 {len(df)} 条销量记录")
        print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"  平均销量: {df['sales_quantity'].mean():.2f}")
        
        # ==================== 步骤 2: 数据预处理 ====================
        print("\n[步骤 2/6] 数据预处理...")
        processor = DataProcessor()
        
        # 创建时间序列数据集
        df = processor.create_time_series_dataset(
            df,
            drug_id=DRUG_ID,
            hospital_id=HOSPITAL_ID
        )
        
        # 处理缺失值和异常值
        df = processor.handle_missing_values(df, method='forward_fill')
        df = processor.handle_outliers(df, 'sales_quantity', method='iqr', threshold=3.0)
        
        print(f"✓ 预处理完成，数据量: {len(df)}")
        
        # ==================== 步骤 3: 特征工程 ====================
        print("\n[步骤 3/6] 特征工程...")
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(
            df,
            target_column='sales_quantity'
        )
        
        print(f"✓ 构建了 {len(df_features.columns)} 个特征")
        print(f"  特征数量: {len(df_features)}")
        
        # 特征列表（排除目标列和日期列）
        feature_cols = [col for col in df_features.columns if col not in ['sales_quantity', 'date']]
        print(f"  可用特征: {len(feature_cols)} 个")
        
        # ==================== 步骤 4: 训练模型 ====================
        print("\n[步骤 4/6] 训练模型...")
        
        # 创建 LightGBM 模型
        model = LightGBMModel()
        
        # 创建训练器
        trainer = ModelTrainer(model, experiment_name="example_run")
        
        # 在完整数据上训练
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
        
        # ==================== 步骤 5: 模型评估 ====================
        print("\n[步骤 5/6] 模型评估...")
        
        # 获取测试集
        split_idx = int(len(df_features) * 0.8)
        test_df = df_features.iloc[split_idx:]
        
        X_test = test_df[feature_cols]
        y_test = test_df['sales_quantity']
        
        # 预测
        y_pred = trained_model.predict(X_test)
        
        # 评估
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test.values, y_pred, return_details=True)
        
        print("✓ 评估完成")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name.upper()}: {metric_value:.4f}")
        
        # 绘制预测结果（可选）
        try:
            print("\n  正在生成可视化图表...")
            evaluator.plot_predictions(
                y_test.values,
                y_pred,
                dates=test_df['date'],
                title=f"{DRUG_ID} - {HOSPITAL_ID} 销量预测",
                save_path=f"examples/prediction_{DRUG_ID}_{HOSPITAL_ID}.png"
            )
            print("  ✓ 图表已保存")
        except Exception as e:
            print(f"  ⚠ 图表生成失败: {e}")
        
        # ==================== 步骤 6: 特征重要性 ====================
        print("\n[步骤 6/6] 特征重要性分析...")
        
        importance_df = trained_model.get_feature_importance()
        
        print("✓ Top 10 重要特征:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.2f}")
        
        # ==================== 保存模型 ====================
        print("\n[可选] 保存模型...")
        model_path = f"models/example_{DRUG_ID}_{HOSPITAL_ID}.txt"
        trained_model.save(model_path)
        print(f"✓ 模型已保存到: {model_path}")
        
        # ==================== 完成 ====================
        print("\n" + "=" * 80)
        print("✅ 示例运行成功完成！")
        print("=" * 80)
        
        print("\n下一步建议:")
        print("  1. 查看 MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")
        print("  2. 尝试批量训练: python scripts/batch_train.py")
        print("  3. 启动 API 服务: uvicorn src.serving.api:app --reload")
        print("  4. 超参数优化: 参考文档中的优化示例")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        print("\n请确保:")
        print("  1. 已安装所有依赖: pip install -r requirements.txt")
        print("  2. 已生成示例数据: python scripts/generate_sample_data.py")
        print("  3. 数据库配置正确: 检查 config/database.yaml")
        sys.exit(1)


if __name__ == "__main__":
    main()
