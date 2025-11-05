"""
完整工作流演示 - 使用 gcode=026436（低钙腹膜透析液）
展示从数据加载到模型训练的完整流程
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

logger = get_logger(__name__)


def main():
    """完整工作流演示"""
    
    print("\n" + "=" * 100)
    print(" 药品销量预测系统 - 完整工作流演示 ".center(100, "="))
    print("=" * 100)
    
    try:
        # 配置参数
        GCODE = "026436"  # 低钙腹膜透析液
        START_DATE = "2020-01-01"  # 使用2020年以后的数据
        END_DATE = "2024-12-31"
        
        print(f"\n📦 目标药品: {GCODE} (低钙腹膜透析液)")
        print(f"📅 数据范围: {START_DATE} 至 {END_DATE}")
        
        # ==================== 步骤 1: 数据加载 ====================
        print("\n" + "─" * 100)
        print("[步骤 1/7] 📥 数据加载")
        print("─" * 100)
        
        loader = DataLoader()
        
        # 加载该药品的所有销售数据（不用 start_date/end_date 参数，避免 dt 分区列过滤问题）
        print(f"正在从数据库加载 gcode={GCODE} 的销售数据...")
        df = loader.load_sales_data(
            gcode=GCODE
        )
        
        # 用 pandas 过滤日期范围
        print(f"过滤日期范围: {START_DATE} 至 {END_DATE}...")
        df = df[(df['create_dt'] >= START_DATE) & (df['create_dt'] <= END_DATE)]
        
        print(f"✅ 成功加载 {len(df)} 条销售记录")
        print(f"   日期范围: {df['create_dt'].min()} 至 {df['create_dt'].max()}")
        print(f"   涉及客户数: {df['cust_name'].nunique()}")
        print(f"   总销量: {df['qty'].sum():,.0f}")
        print(f"   平均销量: {df['qty'].mean():.2f}")
        
        # 显示数据样本
        print(f"\n📊 数据样本（前3条）:")
        print(df[['gcode', 'create_dt', 'qty', 'cust_name', 'gname']].head(3).to_string(index=False))
        
        if len(df) < 100:
            print(f"\n⚠️  警告: 数据量不足 ({len(df)} 条)，需要至少100条数据")
            return
        
        # ==================== 步骤 2: 选择主要客户 ====================
        print("\n" + "─" * 100)
        print("[步骤 2/7] 🏥 选择主要客户")
        print("─" * 100)
        
        # 找出销售记录最多的客户
        customer_stats = df.groupby('cust_name').agg({
            'qty': ['count', 'sum', 'mean']
        }).reset_index()
        customer_stats.columns = ['cust_name', 'record_count', 'total_qty', 'avg_qty']
        customer_stats = customer_stats.sort_values('record_count', ascending=False)
        
        print("📈 销售记录最多的前5个客户:")
        print(customer_stats.head(5).to_string(index=False))
        
        # 选择记录最多的客户
        CUST_NAME = customer_stats.iloc[0]['cust_name']
        print(f"\n✅ 选择客户: {CUST_NAME}")
        print(f"   该客户的销售记录数: {customer_stats.iloc[0]['record_count']:.0f}")
        
        # 筛选该客户的数据
        df_customer = df[df['cust_name'] == CUST_NAME].copy()
        print(f"   筛选后数据量: {len(df_customer)} 条")
        
        # ==================== 步骤 3: 数据预处理 ====================
        print("\n" + "─" * 100)
        print("[步骤 3/7] 🔧 数据预处理")
        print("─" * 100)
        
        processor = DataProcessor()
        
        # 重命名列以适配预处理器
        df_processed = df_customer.rename(columns={
            'create_dt': 'date',
            'qty': 'sales_quantity',
            'gcode': 'drug_id',
            'cust_name': 'hospital_id'
        })
        
        # 确保日期列是 datetime 类型
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values('date').reset_index(drop=True)
        
        # 按日期聚合数据，避免重复的日期标签
        print(f"📊 按日期聚合销售数据...")
        df_processed = df_processed.groupby(['date', 'drug_id', 'hospital_id']).agg({
            'sales_quantity': 'sum'  # 汇总每天的销量
        }).reset_index()
        print(f"   聚合后数据量: {len(df_processed)} 条（每天一条）")
        
        print(f"📅 创建时间序列数据集...")
        df_processed = processor.create_time_series_dataset(
            df_processed,
            drug_id=GCODE,
            hospital_id=CUST_NAME,
            date_column='date',
            target_column='sales_quantity'
        )
        
        print(f"🔍 处理缺失值和异常值...")
        df_processed = processor.handle_missing_values(df_processed, method='forward_fill')
        df_processed = processor.handle_outliers(
            df_processed, 
            'sales_quantity', 
            method='iqr',
            threshold=3.0
        )
        
        print(f"✅ 预处理完成，数据量: {len(df_processed)} 条")
        
        # ==================== 步骤 4: 特征工程 ====================
        print("\n" + "─" * 100)
        print("[步骤 4/7] ⚙️  特征工程")
        print("─" * 100)
        
        print(f"🏗️  构建时间序列特征...")
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(
            df_processed,
            target_column='sales_quantity',
            date_column='date'
        )
        
        print(f"✅ 特征构建完成")
        print(f"   总特征数: {len(df_features.columns)} 个")
        print(f"   特征数据量: {len(df_features)} 条")
        
        # 获取特征列（排除目标列、日期列、标识列，并且只保留数值类型）
        exclude_cols = ['sales_quantity', 'date', 'drug_id', 'hospital_id']
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and df_features[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        print(f"   可用于建模的特征: {len(feature_cols)} 个")
        
        if len(df_features) < 50:
            print(f"\n⚠️  警告: 特征工程后数据量不足 ({len(df_features)} 条)")
            return
        
        # ==================== 步骤 5: 训练模型 ====================
        print("\n" + "─" * 100)
        print("[步骤 5/7] 🤖 训练模型")
        print("─" * 100)
        
        print(f"🚀 使用 LightGBM 训练模型...")
        model = LightGBMModel()
        trainer = ModelTrainer(model, experiment_name="demo_026436")
        
        # 训练模型
        trained_model, test_metrics = trainer.train_on_full_data(
            df_features,
            target_column='sales_quantity',
            test_size=0.2,
            log_mlflow=False
        )
        
        print(f"✅ 模型训练完成！")
        print(f"\n📊 模型性能指标:")
        print(f"   RMSE (均方根误差):  {test_metrics['rmse']:.2f}")
        print(f"   MAE (平均绝对误差):  {test_metrics['mae']:.2f}")
        print(f"   MAPE (平均绝对百分比误差): {test_metrics['mape']:.2f}%")
        print(f"   R² (决定系数):      {test_metrics['r2']:.4f}")
        
        # ==================== 步骤 6: 模型评估 ====================
        print("\n" + "─" * 100)
        print("[步骤 6/7] 📈 模型评估")
        print("─" * 100)
        
        # 在测试集上评估
        split_idx = int(len(df_features) * 0.8)
        test_df = df_features.iloc[split_idx:]
        
        X_test = test_df[feature_cols]
        y_test = test_df['sales_quantity']
        
        print(f"🔍 在测试集上进行预测...")
        y_pred = trained_model.predict(X_test)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test.values, y_pred, return_details=True)
        
        print(f"✅ 评估完成")
        print(f"\n📊 详细评估指标:")
        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name.upper():20s}: {metric_value:.4f}")
        
        # 显示预测样本
        print(f"\n🎯 预测样本对比（最后5条）:")
        comparison_df = pd.DataFrame({
            '实际值': y_test.tail(5).values,
            '预测值': y_pred[-5:],
            '误差': y_test.tail(5).values - y_pred[-5:]
        })
        print(comparison_df.to_string(index=False))
        
        # ==================== 步骤 7: 特征重要性 ====================
        print("\n" + "─" * 100)
        print("[步骤 7/7] 🔍 特征重要性分析")
        print("─" * 100)
        
        importance_df = trained_model.get_feature_importance()
        
        print(f"✅ Top 10 最重要特征:")
        for idx, row in importance_df.head(10).iterrows():
            bar_length = int(row['importance'] / importance_df['importance'].max() * 30)
            bar = "█" * bar_length
            print(f"   {idx+1:2d}. {row['feature']:30s} {bar} {row['importance']:.0f}")
        
        # ==================== 保存模型 ====================
        print("\n" + "─" * 100)
        print("[可选] 💾 保存模型")
        print("─" * 100)
        
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"demo_{GCODE}_{CUST_NAME[:20].replace('/', '_')}.txt"
        
        trained_model.save(str(model_path))
        print(f"✅ 模型已保存到: {model_path}")
        
        # ==================== 完成 ====================
        print("\n" + "=" * 100)
        print(" ✅ 完整工作流演示成功完成！".center(100, "="))
        print("=" * 100)
        
        print(f"\n📊 总结统计:")
        print(f"   药品编码:      {GCODE}")
        print(f"   客户名称:      {CUST_NAME}")
        print(f"   训练数据量:    {len(df_features)} 条")
        print(f"   特征数量:      {len(feature_cols)} 个")
        print(f"   模型性能:      RMSE={test_metrics['rmse']:.2f}, R²={test_metrics['r2']:.4f}")
        print(f"   模型文件:      {model_path}")
        
        print(f"\n🎯 您已了解完整工作流程，包括:")
        print(f"   ✓ 步骤1: 从数据库加载销售数据")
        print(f"   ✓ 步骤2: 选择主要客户进行分析")
        print(f"   ✓ 步骤3: 数据预处理（时间序列、缺失值、异常值）")
        print(f"   ✓ 步骤4: 特征工程（构建时间序列特征）")
        print(f"   ✓ 步骤5: 训练 LightGBM 模型")
        print(f"   ✓ 步骤6: 模型评估和预测")
        print(f"   ✓ 步骤7: 特征重要性分析")
        
        print(f"\n💡 下一步建议:")
        print(f"   1. 尝试其他药品和客户组合")
        print(f"   2. 调整模型超参数优化性能")
        print(f"   3. 使用更长的历史数据进行训练")
        print(f"   4. 批量训练多个药品-客户组合")
        print(f"   5. 启动 API 服务进行在线预测")
        
    except Exception as e:
        logger.error(f"工作流运行失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        print(f"\n请检查:")
        print(f"  1. 数据库连接是否正常")
        print(f"  2. 数据是否充足")
        print(f"  3. 查看日志文件: logs/app.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
