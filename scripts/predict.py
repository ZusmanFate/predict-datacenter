"""
模型预测脚本
使用训练好的模型进行预测
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
from src.utils.logger import get_logger
from src.utils.helpers import get_timestamp

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用模型进行预测')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--drug_id', type=str, required=True,
                        help='药品ID')
    parser.add_argument('--hospital_id', type=str, required=True,
                        help='医院ID')
    parser.add_argument('--start_date', type=str, default=None,
                        help='预测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='预测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='预测结果输出文件')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("开始预测")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"药品ID: {args.drug_id}, 医院ID: {args.hospital_id}")
    logger.info("=" * 80)
    
    try:
        # 1. 加载模型
        logger.info("步骤 1/4: 加载模型")
        model = LightGBMModel()
        model.load(args.model_path)
        logger.info("模型加载完成")
        
        # 2. 加载数据
        logger.info("步骤 2/4: 加载数据")
        loader = DataLoader()
        df = loader.load_sales_data(
            drug_id=args.drug_id,
            hospital_id=args.hospital_id,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if len(df) == 0:
            logger.error("未找到数据")
            return
        
        logger.info(f"加载数据完成，共 {len(df)} 条记录")
        
        # 3. 特征工程
        logger.info("步骤 3/4: 特征工程")
        processor = DataProcessor()
        df = processor.create_time_series_dataset(
            df,
            drug_id=args.drug_id,
            hospital_id=args.hospital_id
        )
        df = processor.handle_missing_values(df, method='forward_fill')
        
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(df, target_column='sales_quantity')
        
        logger.info(f"特征工程完成，共 {len(df_features.columns)} 个特征")
        
        # 4. 预测
        logger.info("步骤 4/4: 预测")
        
        # 获取特征列（排除目标列和日期列）
        feature_cols = [col for col in df_features.columns if col not in ['sales_quantity', 'date']]
        X = df_features[feature_cols]
        
        # 进行预测
        predictions = model.predict(X)
        
        # 创建预测结果 DataFrame
        result_df = pd.DataFrame({
            'date': df_features['date'],
            'actual': df_features['sales_quantity'],
            'predicted': predictions,
            'drug_id': args.drug_id,
            'hospital_id': args.hospital_id
        })
        
        # 计算误差
        result_df['error'] = result_df['actual'] - result_df['predicted']
        result_df['abs_error'] = abs(result_df['error'])
        result_df['percentage_error'] = (result_df['abs_error'] / result_df['actual'] * 100).replace([np.inf, -np.inf], np.nan)
        
        # 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"预测结果已保存到: {output_path}")
        
        # 统计信息
        logger.info("=" * 80)
        logger.info("预测统计:")
        logger.info(f"  记录数: {len(result_df)}")
        logger.info(f"  平均误差: {result_df['error'].mean():.2f}")
        logger.info(f"  平均绝对误差: {result_df['abs_error'].mean():.2f}")
        logger.info(f"  平均百分比误差: {result_df['percentage_error'].mean():.2f}%")
        logger.info(f"  最大预测值: {result_df['predicted'].max():.2f}")
        logger.info(f"  最小预测值: {result_df['predicted'].min():.2f}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"预测过程中出现错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
