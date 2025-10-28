"""
模型训练脚本
命令行工具，用于训练时间序列预测模型
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.models.lgb_model import LightGBMModel
from src.training.trainer import ModelTrainer
from src.utils.logger import get_logger
from src.utils.helpers import get_timestamp

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练药品销量预测模型')
    
    parser.add_argument('--drug_id', type=str, required=True,
                        help='药品ID')
    parser.add_argument('--hospital_id', type=str, required=True,
                        help='医院ID')
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost', 'prophet'],
                        help='模型类型')
    parser.add_argument('--start_date', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='模型输出目录')
    parser.add_argument('--no_mlflow', action='store_true',
                        help='不使用 MLflow 记录')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("开始训练药品销量预测模型")
    logger.info(f"药品ID: {args.drug_id}, 医院ID: {args.hospital_id}")
    logger.info("=" * 80)
    
    try:
        # 1. 加载数据
        logger.info("步骤 1/5: 加载数据")
        loader = DataLoader()
        df = loader.load_sales_data(
            drug_id=args.drug_id,
            hospital_id=args.hospital_id,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if len(df) == 0:
            logger.error("未找到数据，请检查药品ID和医院ID是否正确")
            return
        
        logger.info(f"加载数据完成，共 {len(df)} 条记录，日期范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 2. 数据预处理
        logger.info("步骤 2/5: 数据预处理")
        processor = DataProcessor()
        
        # 创建时间序列数据集
        df = processor.create_time_series_dataset(
            df,
            drug_id=args.drug_id,
            hospital_id=args.hospital_id
        )
        
        # 处理缺失值
        df = processor.handle_missing_values(df, method='forward_fill')
        
        # 处理异常值
        df = processor.handle_outliers(df, 'sales_quantity', method='iqr', threshold=3.0)
        
        logger.info(f"数据预处理完成，最终数据量: {len(df)}")
        
        # 3. 特征工程
        logger.info("步骤 3/5: 特征工程")
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(df, target_column='sales_quantity')
        
        logger.info(f"特征工程完成，共 {len(df_features.columns)} 个特征")
        logger.info(f"特征列表: {df_features.columns.tolist()}")
        
        # 4. 训练模型
        logger.info("步骤 4/5: 训练模型")
        
        if args.model == 'lightgbm':
            model = LightGBMModel()
        else:
            logger.error(f"暂不支持模型类型: {args.model}")
            return
        
        trainer = ModelTrainer(model)
        
        # 在完整数据上训练
        trained_model, test_metrics = trainer.train_on_full_data(
            df_features,
            target_column='sales_quantity',
            test_size=args.test_size,
            log_mlflow=not args.no_mlflow
        )
        
        logger.info("模型训练完成")
        logger.info(f"测试集 RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"测试集 MAE: {test_metrics['mae']:.4f}")
        logger.info(f"测试集 MAPE: {test_metrics['mape']:.4f}%")
        logger.info(f"测试集 R²: {test_metrics['r2']:.4f}")
        
        # 5. 保存模型
        logger.info("步骤 5/5: 保存模型")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = get_timestamp()
        model_filename = f"{args.model}_{args.drug_id}_{args.hospital_id}_{timestamp}.txt"
        model_path = output_dir / model_filename
        
        trained_model.save(str(model_path))
        
        logger.info(f"模型已保存到: {model_path}")
        
        # 保存特征重要性
        importance_df = trained_model.get_feature_importance()
        importance_path = output_dir / f"feature_importance_{args.drug_id}_{args.hospital_id}_{timestamp}.csv"
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        logger.info(f"特征重要性已保存到: {importance_path}")
        
        logger.info("=" * 80)
        logger.info("训练完成！")
        logger.info("=" * 80)
        
        # 显示 Top 10 重要特征
        logger.info("Top 10 重要特征:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.2f}")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
