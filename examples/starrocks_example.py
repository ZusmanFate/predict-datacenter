"""
StarRocks 数据库完整示例
演示从 StarRocks 加载数据到模型训练的完整流程
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
import pandas as pd

logger = get_logger(__name__)


def main():
    """完整示例"""
    
    print("=" * 80)
    print("药品销量预测系统 - StarRocks 数据库示例")
    print("=" * 80)
    
    try:
        # ==================== 步骤 1: 选择数据 ====================
        print("\n[步骤 1/7] 获取药品和客户列表...")
        loader = DataLoader()
        
        # 获取唯一的药品和客户
        print("  正在查询药品列表...")
        gcodes = loader.get_unique_gcodes()
        print(f"  ✓ 找到 {len(gcodes)} 个唯一药品")
        
        print("  正在查询客户列表...")
        cust_names = loader.get_unique_hospitals()
        print(f"  ✓ 找到 {len(cust_names)} 个唯一客户")
        
        if not gcodes or not cust_names:
            logger.error("未找到药品或客户数据")
            return
        
        # 过滤掉 None 和空字符串，选择有效的药品和客户
        valid_gcodes = [g for g in gcodes if g and str(g).strip() and str(g) != 'None']
        valid_cust_names = [c for c in cust_names if c and str(c).strip() and str(c) != 'None']
        
        if not valid_gcodes or not valid_cust_names:
            logger.error("未找到有效的药品或客户数据")
            return
        
        # 选择第一个有效的药品和客户作为示例
        GCODE = valid_gcodes[0]
        CUST_NAME = valid_cust_names[0]
        
        print(f"\n  选择示例组合:")
        print(f"    药品编码: {GCODE}")
        print(f"    客户名称: {CUST_NAME}")
        
        # ==================== 步骤 2: 加载数据 ====================
        print("\n[步骤 2/7] 加载销量数据...")
        
        # 加载最近的数据（限制数量用于快速测试）
        df = loader.load_sales_data(
            gcode=GCODE,
            cust_name=CUST_NAME,
            limit=2000  # 限制数据量用于快速测试
        )
        
        if len(df) < 100:
            logger.error(f"数据量不足: {len(df)} 条，至少需要 100 条")
            print(f"\n  ⚠ 数据量不足，尝试其他药品-客户组合...")
            
            # 尝试找一个数据量足够的组合
            for gcode in gcodes[:10]:
                for cust_name in cust_names[:5]:
                    df_test = loader.load_sales_data(
                        gcode=gcode,
                        cust_name=cust_name,
                        limit=2000
                    )
                    if len(df_test) >= 100:
                        GCODE = gcode
                        CUST_NAME = cust_name
                        df = df_test
                        print(f"  ✓ 找到合适的组合: {GCODE} - {CUST_NAME}")
                        break
                if len(df) >= 100:
                    break
            
            if len(df) < 100:
                logger.error("未找到数据量足够的组合")
                return
        
        print(f"  ✓ 加载了 {len(df)} 条销量记录")
        print(f"  日期范围: {df['create_dt'].min()} 到 {df['create_dt'].max()}")
        print(f"  平均销量: {df['qty'].mean():.2f}")
        print(f"  总销量: {df['qty'].sum():.0f}")
        
        # ==================== 步骤 3: 数据预处理 ====================
        print("\n[步骤 3/7] 数据预处理...")
        processor = DataProcessor()
        
        # 重命名列以适配预处理器（保持向后兼容）
        df_processed = df.rename(columns={
            'create_dt': 'date',
            'qty': 'sales_quantity',
            'gcode': 'drug_id',
            'cust_name': 'hospital_id'
        })
        
        # 确保日期列是 datetime 类型
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        # 按日期排序
        df_processed = df_processed.sort_values('date').reset_index(drop=True)
        
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
        df_processed = processor.handle_outliers(
            df_processed, 
            'sales_quantity', 
            method='iqr',
            threshold=3.0
        )
        
        print(f"  ✓ 预处理完成，数据量: {len(df_processed)}")
        
        # ==================== 步骤 4: 特征工程 ====================
        print("\n[步骤 4/7] 特征工程...")
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(
            df_processed,
            target_column='sales_quantity',
            date_column='date'
        )
        
        print(f"  ✓ 构建了 {len(df_features.columns)} 个特征")
        print(f"  特征数据量: {len(df_features)} 条")
        
        # 获取特征列（排除目标列和日期列）
        feature_cols = [col for col in df_features.columns 
                       if col not in ['sales_quantity', 'date', 'drug_id', 'hospital_id']]
        print(f"  可用特征: {len(feature_cols)} 个")
        
        if len(df_features) < 50:
            logger.error(f"特征工程后数据量不足: {len(df_features)} 条")
            return
        
        # ==================== 步骤 5: 训练模型 ====================
        print("\n[步骤 5/7] 训练模型...")
        
        # 创建 LightGBM 模型
        model = LightGBMModel()
        
        # 创建训练器
        trainer = ModelTrainer(model, experiment_name="starrocks_example")
        
        # 在完整数据上训练
        trained_model, test_metrics = trainer.train_on_full_data(
            df_features,
            target_column='sales_quantity',
            test_size=0.2,
            log_mlflow=False  # 暂时不记录到 MLflow
        )
        
        print("  ✓ 模型训练完成")
        print(f"    RMSE: {test_metrics['rmse']:.4f}")
        print(f"    MAE: {test_metrics['mae']:.4f}")
        print(f"    MAPE: {test_metrics['mape']:.4f}%")
        print(f"    R²: {test_metrics['r2']:.4f}")
        
        # ==================== 步骤 6: 模型评估 ====================
        print("\n[步骤 6/7] 模型评估...")
        
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
        
        print("  ✓ 评估完成")
        for metric_name, metric_value in metrics.items():
            print(f"    {metric_name.upper()}: {metric_value:.4f}")
        
        # ==================== 步骤 7: 特征重要性 ====================
        print("\n[步骤 7/7] 特征重要性分析...")
        
        importance_df = trained_model.get_feature_importance()
        
        print("  ✓ Top 10 重要特征:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"    {idx+1}. {row['feature']}: {row['importance']:.2f}")
        
        # ==================== 保存模型 ====================
        print("\n[可选] 保存模型...")
        model_path = f"models/starrocks_{GCODE}_{CUST_NAME[:20]}.txt"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        trained_model.save(model_path)
        print(f"  ✓ 模型已保存到: {model_path}")
        
        # ==================== 完成 ====================
        print("\n" + "=" * 80)
        print("✅ 示例运行成功完成！")
        print("=" * 80)
        
        print("\n📊 数据统计:")
        print(f"  药品编码: {GCODE}")
        print(f"  客户名称: {CUST_NAME}")
        print(f"  训练数据: {len(df_features)} 条")
        print(f"  特征数量: {len(feature_cols)} 个")
        print(f"  模型性能: RMSE={test_metrics['rmse']:.2f}, R²={test_metrics['r2']:.4f}")
        
        print("\n🎯 下一步建议:")
        print("  1. 尝试其他药品和客户组合")
        print("  2. 调整特征工程参数: config/config.yaml")
        print("  3. 批量训练多个模型: python scripts/batch_train.py")
        print("  4. 启动 API 服务: uvicorn src.serving.api:app --reload")
        print("  5. 超参数优化: 使用 HyperparameterOptimizer")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("  1. StarRocks 连接是否正常")
        print("  2. 数据表是否有足够的数据")
        print("  3. 查看日志文件: logs/app.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
