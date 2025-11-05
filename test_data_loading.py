"""
测试数据加载功能
从 StarRocks 数据库加载少量数据进行测试
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_data_loading():
    """测试数据加载"""
    try:
        logger.info("=" * 60)
        logger.info("测试数据加载...")
        logger.info("=" * 60)
        
        loader = DataLoader()
        
        # 获取唯一的药品和医院列表
        logger.info("1. 获取药品和医院列表...")
        gcodes = loader.get_unique_gcodes()
        logger.info(f"   ✓ 找到 {len(gcodes)} 个唯一药品")
        logger.info(f"   前5个药品: {gcodes[:5]}")
        
        cust_names = loader.get_unique_hospitals()
        logger.info(f"   ✓ 找到 {len(cust_names)} 个唯一客户")
        logger.info(f"   前5个客户: {cust_names[:5]}")
        
        # 加载单个药品-医院的数据
        if gcodes and cust_names:
            logger.info("\n2. 加载示例数据...")
            gcode = gcodes[0]
            cust_name = cust_names[0]
            
            logger.info(f"   药品: {gcode}")
            logger.info(f"   客户: {cust_name}")
            
            df = loader.load_sales_data(
                gcode=gcode,
                cust_name=cust_name,
                limit=100  # 只加载100条测试
            )
            
            logger.info(f"   ✓ 成功加载 {len(df)} 条数据")
            logger.info(f"   数据列: {df.columns.tolist()}")
            
            if len(df) > 0:
                logger.info(f"   日期范围: {df['create_dt'].min()} 到 {df['create_dt'].max()}")
                logger.info(f"   平均销量: {df['qty'].mean():.2f}")
            
            logger.info("\n数据样本:")
            print(df.head())
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 数据加载测试完成！")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
