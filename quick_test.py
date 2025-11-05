"""
快速测试脚本 - 一键测试所有核心功能
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.database import get_db_manager
from src.data.loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_database_connection():
    """测试1：数据库连接"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 1/4: 数据库连接")
    logger.info("=" * 60)
    
    try:
        db_manager = get_db_manager()
        logger.info(f"✓ 数据库类型: {db_manager.db_type}")
        logger.info(f"✓ 连接成功")
        return True
    except Exception as e:
        logger.error(f"✗ 连接失败: {e}")
        return False


def test_data_loading():
    """测试2：数据加载"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2/4: 数据加载")
    logger.info("=" * 60)
    
    try:
        loader = DataLoader()
        
        # 获取药品列表
        gcodes = loader.get_unique_gcodes()
        logger.info(f"✓ 找到 {len(gcodes)} 个药品")
        
        # 获取客户列表
        cust_names = loader.get_unique_hospitals()
        logger.info(f"✓ 找到 {len(cust_names)} 个客户")
        
        if gcodes and cust_names:
            # 加载示例数据
            gcode = gcodes[0]
            cust_name = cust_names[0]
            
            df = loader.load_sales_data(
                gcode=gcode,
                cust_name=cust_name,
                limit=10
            )
            
            logger.info(f"✓ 成功加载数据: {len(df)} 条")
            logger.info(f"  示例药品: {gcode}")
            logger.info(f"  示例客户: {cust_name}")
            
            return True
        else:
            logger.warning("⚠ 未找到数据")
            return False
            
    except Exception as e:
        logger.error(f"✗ 数据加载失败: {e}")
        return False


def test_feature_engineering():
    """测试3：特征工程"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3/4: 特征工程")
    logger.info("=" * 60)
    
    try:
        from src.data.loader import DataLoader
        from src.data.processor import DataProcessor
        from src.features.builder import FeatureBuilder
        
        loader = DataLoader()
        gcodes = loader.get_unique_gcodes()
        cust_names = loader.get_unique_hospitals()
        
        if not gcodes or not cust_names:
            logger.warning("⚠ 无数据可测试")
            return False
        
        # 加载数据
        df = loader.load_sales_data(
            gcode=gcodes[0],
            cust_name=cust_names[0],
            limit=100
        )
        
        if len(df) < 50:
            logger.warning(f"⚠ 数据量不足: {len(df)} 条")
            return False
        
        # 预处理
        processor = DataProcessor()
        df_processed = df.rename(columns={
            'create_dt': 'date',
            'qty': 'sales_quantity',
            'gcode': 'drug_id',
            'cust_name': 'hospital_id'
        })
        
        df_processed = processor.create_time_series_dataset(
            df_processed,
            drug_id=gcodes[0],
            hospital_id=cust_names[0]
        )
        
        # 特征工程
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(df_processed)
        
        logger.info(f"✓ 特征构建成功")
        logger.info(f"  原始数据: {len(df)} 条")
        logger.info(f"  特征数据: {len(df_features)} 条")
        logger.info(f"  特征数量: {len(df_features.columns)} 个")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 特征工程失败: {e}")
        return False


def test_model_components():
    """测试4：模型组件"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 4/4: 模型组件")
    logger.info("=" * 60)
    
    try:
        from src.models.lgb_model import LightGBMModel
        from src.training.trainer import ModelTrainer
        from src.training.evaluator import ModelEvaluator
        
        # 测试模型初始化
        model = LightGBMModel()
        logger.info(f"✓ LightGBM 模型初始化成功")
        
        # 测试训练器
        trainer = ModelTrainer(model)
        logger.info(f"✓ 训练器初始化成功")
        
        # 测试评估器
        evaluator = ModelEvaluator()
        logger.info(f"✓ 评估器初始化成功")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 模型组件测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    logger.info("\n" + "=" * 80)
    logger.info("🚀 药品销量预测系统 - 快速测试")
    logger.info("=" * 80)
    
    results = []
    
    # 运行所有测试
    results.append(("数据库连接", test_database_connection()))
    results.append(("数据加载", test_data_loading()))
    results.append(("特征工程", test_feature_engineering()))
    results.append(("模型组件", test_model_components()))
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        logger.info(f"{status} - {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    logger.info(f"\n通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.0f}%)")
    
    if passed_count == total_count:
        logger.info("\n" + "=" * 80)
        logger.info("🎉 所有测试通过！系统运行正常")
        logger.info("=" * 80)
        logger.info("\n下一步：")
        logger.info("  1. 运行完整示例: python examples/impala_example.py")
        logger.info("  2. 查看详细指南: RUN_GUIDE.md")
        logger.info("  3. 启动 API 服务: uvicorn src.serving.api:app --reload")
    else:
        logger.warning("\n" + "=" * 80)
        logger.warning("⚠ 部分测试失败，请检查配置和依赖")
        logger.warning("=" * 80)
        logger.info("\n故障排查：")
        logger.info("  1. 检查数据库配置: config/database.yaml")
        logger.info("  2. 查看日志文件: logs/app.log")
        logger.info("  3. 参考运行指南: RUN_GUIDE.md")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
