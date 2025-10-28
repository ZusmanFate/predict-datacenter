"""
生成示例数据
用于快速测试和演示
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database import get_db_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_sales_data(
    drug_ids: list,
    hospital_ids: list,
    start_date: str = '2022-01-01',
    end_date: str = '2024-12-31',
    base_sales: int = 100,
    trend: float = 0.01,
    seasonality: bool = True,
    noise_level: float = 0.2
) -> pd.DataFrame:
    """
    生成模拟销量数据
    
    Args:
        drug_ids: 药品ID列表
        hospital_ids: 医院ID列表
        start_date: 开始日期
        end_date: 结束日期
        base_sales: 基础销量
        trend: 趋势系数
        seasonality: 是否包含季节性
        noise_level: 噪声水平
        
    Returns:
        销量数据 DataFrame
    """
    logger.info(f"生成示例数据: {len(drug_ids)} 个药品 × {len(hospital_ids)} 个医院")
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    
    for drug_id in drug_ids:
        for hospital_id in hospital_ids:
            # 为每个药品-医院组合生成不同的参数
            drug_base = base_sales * np.random.uniform(0.5, 2.0)
            drug_trend = trend * np.random.uniform(0.5, 1.5)
            
            for idx, date in enumerate(date_range):
                # 趋势成分
                trend_component = drug_base * (1 + drug_trend * idx / len(date_range))
                
                # 季节性成分
                if seasonality:
                    # 年度季节性
                    seasonal_component = 20 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                    # 周季节性（周末销量下降）
                    weekly_component = -10 if date.dayofweek >= 5 else 5
                else:
                    seasonal_component = 0
                    weekly_component = 0
                
                # 噪声
                noise = np.random.normal(0, drug_base * noise_level)
                
                # 总销量
                sales_quantity = max(0, trend_component + seasonal_component + weekly_component + noise)
                
                # 价格（带一些随机波动）
                price = np.random.uniform(10, 50)
                sales_amount = sales_quantity * price
                
                data.append({
                    'drug_id': drug_id,
                    'drug_name': f'药品_{drug_id}',
                    'hospital_id': hospital_id,
                    'hospital_name': f'医院_{hospital_id}',
                    'date': date,
                    'sales_quantity': round(sales_quantity, 2),
                    'sales_amount': round(sales_amount, 2),
                    'price': round(price, 2),
                    'category': f'类别{np.random.randint(1, 6)}',
                    'region': f'区域{np.random.randint(1, 4)}'
                })
    
    df = pd.DataFrame(data)
    logger.info(f"生成完成，共 {len(df)} 条记录")
    return df


def generate_drug_info(drug_ids: list) -> pd.DataFrame:
    """生成药品信息"""
    data = []
    for drug_id in drug_ids:
        data.append({
            'drug_id': drug_id,
            'drug_name': f'药品_{drug_id}',
            'category': f'类别{np.random.randint(1, 6)}',
            'manufacturer': f'厂商{np.random.randint(1, 11)}',
            'specification': f'{np.random.choice([10, 20, 50, 100])}mg'
        })
    return pd.DataFrame(data)


def generate_hospital_info(hospital_ids: list) -> pd.DataFrame:
    """生成医院信息"""
    data = []
    for hospital_id in hospital_ids:
        data.append({
            'hospital_id': hospital_id,
            'hospital_name': f'医院_{hospital_id}',
            'region': f'区域{np.random.randint(1, 4)}',
            'level': np.random.choice(['三甲', '三乙', '二甲']),
            'type': np.random.choice(['综合医院', '专科医院', '中医医院'])
        })
    return pd.DataFrame(data)


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始生成示例数据")
    logger.info("=" * 80)
    
    # 配置
    drug_ids = [f'D{i:03d}' for i in range(1, 11)]  # 10个药品
    hospital_ids = [f'H{i:03d}' for i in range(1, 6)]  # 5个医院
    
    try:
        # 获取数据库管理器
        db_manager = get_db_manager()
        
        # 创建数据表
        logger.info("创建数据表...")
        db_manager.create_tables()
        
        # 生成销量数据
        logger.info("生成销量数据...")
        sales_df = generate_sales_data(
            drug_ids=drug_ids,
            hospital_ids=hospital_ids,
            start_date='2022-01-01',
            end_date='2024-12-31'
        )
        
        # 生成药品信息
        logger.info("生成药品信息...")
        drugs_df = generate_drug_info(drug_ids)
        
        # 生成医院信息
        logger.info("生成医院信息...")
        hospitals_df = generate_hospital_info(hospital_ids)
        
        # 写入数据库
        logger.info("写入数据库...")
        
        sales_table = db_manager.config['tables']['sales']['name']
        drugs_table = db_manager.config['tables']['drugs']['name']
        hospitals_table = db_manager.config['tables']['hospitals']['name']
        
        sales_df.to_sql(sales_table, db_manager.engine, if_exists='replace', index=False)
        drugs_df.to_sql(drugs_table, db_manager.engine, if_exists='replace', index=False)
        hospitals_df.to_sql(hospitals_table, db_manager.engine, if_exists='replace', index=False)
        
        logger.info("数据写入完成")
        
        # 同时保存到 CSV（用于备份和查看）
        data_dir = Path('data/raw')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        sales_df.to_csv(data_dir / 'sales_data.csv', index=False, encoding='utf-8-sig')
        drugs_df.to_csv(data_dir / 'drug_info.csv', index=False, encoding='utf-8-sig')
        hospitals_df.to_csv(data_dir / 'hospital_info.csv', index=False, encoding='utf-8-sig')
        
        logger.info(f"CSV 文件已保存到: {data_dir}")
        
        # 数据统计
        logger.info("=" * 80)
        logger.info("数据统计:")
        logger.info(f"  药品数量: {len(drug_ids)}")
        logger.info(f"  医院数量: {len(hospital_ids)}")
        logger.info(f"  销量记录: {len(sales_df)}")
        logger.info(f"  日期范围: {sales_df['date'].min()} 到 {sales_df['date'].max()}")
        logger.info(f"  平均销量: {sales_df['sales_quantity'].mean():.2f}")
        logger.info("=" * 80)
        
        # 示例数据预览
        logger.info("\n示例数据（前5条）:")
        print(sales_df.head())
        
        logger.info("\n✅ 示例数据生成完成！")
        logger.info("\n下一步:")
        logger.info("  1. 训练模型: python scripts/train.py --drug_id D001 --hospital_id H001")
        logger.info("  2. 启动API: uvicorn src.serving.api:app --host 0.0.0.0 --port 8000")
        
    except Exception as e:
        logger.error(f"生成示例数据失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
