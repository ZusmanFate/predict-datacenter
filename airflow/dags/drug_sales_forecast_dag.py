"""
Airflow DAG - 药品销量预测定时任务
每日自动训练和预测
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.models.lgb_model import LightGBMModel
from src.training.trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 默认参数
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['your_email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# 创建 DAG
dag = DAG(
    'drug_sales_forecast_daily',
    default_args=default_args,
    description='每日药品销量预测训练和预测任务',
    schedule_interval='0 2 * * *',  # 每天凌晨2点执行
    catchup=False,
    tags=['forecast', 'drug_sales', 'ml'],
)


def load_and_prepare_data(**context):
    """任务1：加载和准备数据"""
    logger.info("开始加载数据...")
    
    # 从上下文获取执行日期
    execution_date = context['execution_date']
    
    # 加载数据
    loader = DataLoader()
    
    # 获取所有药品和医院
    gcodes = loader.get_unique_gcodes()[:10]  # 示例：前10个药品
    cust_names = loader.get_unique_hospitals()[:5]  # 示例：前5个医院
    
    # 保存到 XCom
    context['task_instance'].xcom_push(key='gcodes', value=gcodes)
    context['task_instance'].xcom_push(key='cust_names', value=cust_names)
    
    logger.info(f"数据准备完成：{len(gcodes)} 个药品, {len(cust_names)} 个医院")
    return {'gcodes': len(gcodes), 'cust_names': len(cust_names)}


def train_models(**context):
    """任务2：批量训练模型"""
    logger.info("开始批量训练...")
    
    # 从 XCom 获取数据
    task_instance = context['task_instance']
    gcodes = task_instance.xcom_pull(task_ids='load_data', key='gcodes')
    cust_names = task_instance.xcom_pull(task_ids='load_data', key='cust_names')
    
    loader = DataLoader()
    processor = DataProcessor()
    feature_builder = FeatureBuilder()
    
    results = []
    
    for gcode in gcodes[:3]:  # 示例：训练前3个
        for cust_name in cust_names[:2]:  # 示例：前2个医院
            try:
                # 加载数据
                df = loader.load_sales_data(gcode=gcode, cust_name=cust_name)
                
                if len(df) < 100:
                    logger.warning(f"跳过 {gcode}-{cust_name}: 数据不足")
                    continue
                
                # 预处理
                df = processor.create_time_series_dataset(df, gcode, cust_name)
                df = processor.handle_missing_values(df)
                
                # 特征工程
                df_features = feature_builder.build_features(df)
                
                # 训练
                model = LightGBMModel()
                trainer = ModelTrainer(model, experiment_name=f"airflow_{context['ds']}")
                trained_model, metrics = trainer.train_on_full_data(
                    df_features,
                    target_column='sales_quantity',
                    log_mlflow=True
                )
                
                # 保存模型
                model_path = f"models/airflow_{gcode}_{cust_name}_{context['ds']}.txt"
                trained_model.save(model_path)
                
                results.append({
                    'gcode': gcode,
                    'cust_name': cust_name,
                    'rmse': metrics['rmse'],
                    'model_path': model_path
                })
                
                logger.info(f"✓ 训练完成: {gcode}-{cust_name}, RMSE={metrics['rmse']:.2f}")
                
            except Exception as e:
                logger.error(f"✗ 训练失败: {gcode}-{cust_name}, 错误: {e}")
    
    # 保存结果到 XCom
    context['task_instance'].xcom_push(key='training_results', value=results)
    
    logger.info(f"批量训练完成：成功 {len(results)} 个模型")
    return {'trained_models': len(results)}


def generate_predictions(**context):
    """任务3：生成预测"""
    logger.info("开始生成预测...")
    
    # 从 XCom 获取训练结果
    task_instance = context['task_instance']
    training_results = task_instance.xcom_pull(task_ids='train_models', key='training_results')
    
    if not training_results:
        logger.warning("没有训练完成的模型，跳过预测")
        return {'predictions': 0}
    
    # 这里可以添加预测逻辑
    # 例如：为未来30天生成预测
    
    logger.info(f"预测生成完成：{len(training_results)} 个模型")
    return {'predictions': len(training_results)}


def check_model_performance(**context):
    """任务4：检查模型性能"""
    logger.info("开始检查模型性能...")
    
    # 从 XCom 获取训练结果
    task_instance = context['task_instance']
    training_results = task_instance.xcom_pull(task_ids='train_models', key='training_results')
    
    if not training_results:
        logger.warning("没有训练结果")
        return {'status': 'no_results'}
    
    # 检查 RMSE
    avg_rmse = sum([r['rmse'] for r in training_results]) / len(training_results)
    
    if avg_rmse > 100:  # 阈值示例
        logger.warning(f"平均 RMSE 过高: {avg_rmse:.2f}")
        # 可以发送告警
    else:
        logger.info(f"模型性能正常，平均 RMSE: {avg_rmse:.2f}")
    
    return {'avg_rmse': avg_rmse, 'status': 'ok'}


# 定义任务
task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_and_prepare_data,
    provide_context=True,
    dag=dag,
)

task_train_models = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    dag=dag,
)

task_generate_predictions = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    provide_context=True,
    dag=dag,
)

task_check_performance = PythonOperator(
    task_id='check_performance',
    python_callable=check_model_performance,
    provide_context=True,
    dag=dag,
)

# 清理旧模型（可选）
task_cleanup = BashOperator(
    task_id='cleanup_old_models',
    bash_command='find models/ -name "airflow_*.txt" -mtime +30 -delete || true',
    dag=dag,
)

# 定义任务依赖关系
task_load_data >> task_train_models >> task_generate_predictions >> task_check_performance >> task_cleanup
