"""
批量训练脚本
为多个药品-医院组合批量训练模型
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
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
    parser = argparse.ArgumentParser(description='批量训练药品销量预测模型')
    
    parser.add_argument('--drug_ids', type=str, nargs='+', default=None,
                        help='药品ID列表（留空表示所有药品）')
    parser.add_argument('--hospital_ids', type=str, nargs='+', default=None,
                        help='医院ID列表（留空表示所有医院）')
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost', 'prophet'],
                        help='模型类型')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='并行训练的最大线程数')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='模型输出目录')
    parser.add_argument('--min_data_points', type=int, default=100,
                        help='最小数据点数（少于此数量跳过）')
    parser.add_argument('--save_results', type=str, default='batch_results.csv',
                        help='批量训练结果保存路径')
    
    return parser.parse_args()


def train_single_model(
    drug_id: str,
    hospital_id: str,
    model_type: str,
    test_size: float,
    output_dir: str,
    min_data_points: int
) -> dict:
    """
    训练单个药品-医院组合的模型
    
    Args:
        drug_id: 药品ID
        hospital_id: 医院ID
        model_type: 模型类型
        test_size: 测试集比例
        output_dir: 输出目录
        min_data_points: 最小数据点数
        
    Returns:
        训练结果字典
    """
    result = {
        'drug_id': drug_id,
        'hospital_id': hospital_id,
        'status': 'failed',
        'message': '',
        'data_points': 0,
        'rmse': None,
        'mae': None,
        'mape': None,
        'r2': None,
        'model_path': None
    }
    
    try:
        # 1. 加载数据
        loader = DataLoader()
        df = loader.load_sales_data(drug_id=drug_id, hospital_id=hospital_id)
        
        if len(df) < min_data_points:
            result['message'] = f'数据点不足 ({len(df)} < {min_data_points})'
            result['data_points'] = len(df)
            logger.warning(f"跳过 {drug_id}-{hospital_id}: {result['message']}")
            return result
        
        result['data_points'] = len(df)
        
        # 2. 数据预处理
        processor = DataProcessor()
        df = processor.create_time_series_dataset(df, drug_id=drug_id, hospital_id=hospital_id)
        df = processor.handle_missing_values(df, method='forward_fill')
        df = processor.handle_outliers(df, 'sales_quantity', method='iqr', threshold=3.0)
        
        # 3. 特征工程
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(df, target_column='sales_quantity')
        
        if len(df_features) < min_data_points // 2:
            result['message'] = f'特征工程后数据不足 ({len(df_features)})'
            logger.warning(f"跳过 {drug_id}-{hospital_id}: {result['message']}")
            return result
        
        # 4. 训练模型
        if model_type == 'lightgbm':
            model = LightGBMModel()
        else:
            result['message'] = f'不支持的模型类型: {model_type}'
            return result
        
        trainer = ModelTrainer(model, experiment_name=f"batch_training_{get_timestamp()}")
        trained_model, test_metrics = trainer.train_on_full_data(
            df_features,
            target_column='sales_quantity',
            test_size=test_size,
            log_mlflow=False  # 批量训练时不记录 MLflow（避免过多记录）
        )
        
        # 5. 保存模型
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_filename = f"{model_type}_{drug_id}_{hospital_id}.txt"
        model_path = output_path / model_filename
        trained_model.save(str(model_path))
        
        # 6. 更新结果
        result['status'] = 'success'
        result['message'] = '训练成功'
        result['rmse'] = test_metrics['rmse']
        result['mae'] = test_metrics['mae']
        result['mape'] = test_metrics['mape']
        result['r2'] = test_metrics['r2']
        result['model_path'] = str(model_path)
        
        logger.info(f"✅ 完成 {drug_id}-{hospital_id}: RMSE={test_metrics['rmse']:.2f}")
        
    except Exception as e:
        result['message'] = str(e)
        logger.error(f"❌ 失败 {drug_id}-{hospital_id}: {e}")
    
    return result


def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("开始批量训练")
    logger.info("=" * 80)
    
    try:
        # 获取药品和医院列表
        loader = DataLoader()
        
        if args.drug_ids:
            drug_ids = args.drug_ids
        else:
            drug_ids = loader.get_unique_drugs()
            logger.info(f"自动获取所有药品: {len(drug_ids)} 个")
        
        if args.hospital_ids:
            hospital_ids = args.hospital_ids
        else:
            hospital_ids = loader.get_unique_hospitals()
            logger.info(f"自动获取所有医院: {len(hospital_ids)} 个")
        
        # 生成所有组合
        combinations = [(drug_id, hospital_id) for drug_id in drug_ids for hospital_id in hospital_ids]
        total_combinations = len(combinations)
        
        logger.info(f"总计组合数: {total_combinations}")
        logger.info(f"并行线程数: {args.max_workers}")
        
        # 批量训练
        results = []
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            future_to_combo = {
                executor.submit(
                    train_single_model,
                    drug_id,
                    hospital_id,
                    args.model,
                    args.test_size,
                    args.output_dir,
                    args.min_data_points
                ): (drug_id, hospital_id)
                for drug_id, hospital_id in combinations
            }
            
            # 使用 tqdm 显示进度
            with tqdm(total=total_combinations, desc="训练进度") as pbar:
                for future in as_completed(future_to_combo):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        
        # 汇总结果
        results_df = pd.DataFrame(results)
        
        # 保存结果
        results_path = Path(args.save_results)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        logger.info("=" * 80)
        logger.info("批量训练完成！")
        logger.info("=" * 80)
        
        # 统计信息
        success_count = (results_df['status'] == 'success').sum()
        failed_count = (results_df['status'] == 'failed').sum()
        
        logger.info(f"总组合数: {total_combinations}")
        logger.info(f"成功: {success_count}")
        logger.info(f"失败: {failed_count}")
        logger.info(f"成功率: {success_count/total_combinations*100:.2f}%")
        
        if success_count > 0:
            logger.info("\n性能统计（成功的模型）:")
            success_results = results_df[results_df['status'] == 'success']
            logger.info(f"  平均 RMSE: {success_results['rmse'].mean():.4f}")
            logger.info(f"  平均 MAE: {success_results['mae'].mean():.4f}")
            logger.info(f"  平均 MAPE: {success_results['mape'].mean():.4f}%")
            logger.info(f"  平均 R²: {success_results['r2'].mean():.4f}")
        
        logger.info(f"\n结果已保存到: {results_path}")
        
        # 显示失败的组合
        if failed_count > 0:
            logger.info("\n失败的组合:")
            failed_results = results_df[results_df['status'] == 'failed']
            for idx, row in failed_results.iterrows():
                logger.info(f"  {row['drug_id']}-{row['hospital_id']}: {row['message']}")
        
    except Exception as e:
        logger.error(f"批量训练过程中出现错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
