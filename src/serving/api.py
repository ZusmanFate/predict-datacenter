"""
FastAPI 预测服务
提供 REST API 接口进行预测
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.models.lgb_model import LightGBMModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="药品销量预测 API",
    description="时间序列预测服务 - 药品销量预测",
    version="1.0.0"
)

# 全局模型缓存
model_cache = {}


class PredictionRequest(BaseModel):
    """单次预测请求"""
    drug_id: str = Field(..., description="药品ID")
    hospital_id: str = Field(..., description="医院ID")
    model_path: str = Field(..., description="模型文件路径")
    start_date: Optional[str] = Field(None, description="开始日期 (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="结束日期 (YYYY-MM-DD)")


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    drug_ids: List[str] = Field(..., description="药品ID列表")
    hospital_ids: List[str] = Field(..., description="医院ID列表")
    model_path: str = Field(..., description="模型文件路径")
    start_date: Optional[str] = Field(None, description="开始日期")
    end_date: Optional[str] = Field(None, description="结束日期")


class PredictionResponse(BaseModel):
    """预测响应"""
    drug_id: str
    hospital_id: str
    predictions: List[dict]
    summary: dict


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "药品销量预测 API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


def load_model(model_path: str) -> LightGBMModel:
    """
    加载模型（带缓存）
    
    Args:
        model_path: 模型路径
        
    Returns:
        模型实例
    """
    if model_path not in model_cache:
        logger.info(f"加载模型: {model_path}")
        model = LightGBMModel()
        model.load(model_path)
        model_cache[model_path] = model
    else:
        logger.info(f"使用缓存的模型: {model_path}")
    
    return model_cache[model_path]


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    单次预测
    
    Args:
        request: 预测请求
        
    Returns:
        预测结果
    """
    try:
        logger.info(f"收到预测请求: drug_id={request.drug_id}, hospital_id={request.hospital_id}")
        
        # 1. 加载模型
        model = load_model(request.model_path)
        
        # 2. 加载数据
        loader = DataLoader()
        df = loader.load_sales_data(
            drug_id=request.drug_id,
            hospital_id=request.hospital_id,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="未找到数据")
        
        # 3. 数据预处理和特征工程
        processor = DataProcessor()
        df = processor.create_time_series_dataset(
            df,
            drug_id=request.drug_id,
            hospital_id=request.hospital_id
        )
        df = processor.handle_missing_values(df, method='forward_fill')
        
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(df, target_column='sales_quantity')
        
        # 4. 预测
        feature_cols = [col for col in df_features.columns if col not in ['sales_quantity', 'date']]
        X = df_features[feature_cols]
        predictions = model.predict(X)
        
        # 5. 构建响应
        prediction_list = []
        for idx, pred in enumerate(predictions):
            prediction_list.append({
                "date": str(df_features['date'].iloc[idx]),
                "predicted": float(pred),
                "actual": float(df_features['sales_quantity'].iloc[idx])
            })
        
        # 统计摘要
        summary = {
            "total_records": len(predictions),
            "avg_prediction": float(predictions.mean()),
            "max_prediction": float(predictions.max()),
            "min_prediction": float(predictions.min()),
            "date_range": {
                "start": str(df_features['date'].min()),
                "end": str(df_features['date'].max())
            }
        }
        
        logger.info(f"预测完成: {len(predictions)} 条记录")
        
        return PredictionResponse(
            drug_id=request.drug_id,
            hospital_id=request.hospital_id,
            predictions=prediction_list,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """
    批量预测
    
    Args:
        request: 批量预测请求
        
    Returns:
        批量预测结果
    """
    try:
        logger.info(f"收到批量预测请求: {len(request.drug_ids)} 个药品, {len(request.hospital_ids)} 个医院")
        
        results = []
        
        # 对每个药品-医院组合进行预测
        for drug_id in request.drug_ids:
            for hospital_id in request.hospital_ids:
                try:
                    pred_request = PredictionRequest(
                        drug_id=drug_id,
                        hospital_id=hospital_id,
                        model_path=request.model_path,
                        start_date=request.start_date,
                        end_date=request.end_date
                    )
                    result = await predict(pred_request)
                    results.append(result.dict())
                except Exception as e:
                    logger.warning(f"预测失败 ({drug_id}, {hospital_id}): {e}")
                    results.append({
                        "drug_id": drug_id,
                        "hospital_id": hospital_id,
                        "error": str(e)
                    })
        
        logger.info(f"批量预测完成: {len(results)} 个结果")
        
        return {
            "total_predictions": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"批量预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


@app.get("/models")
async def list_models():
    """列出已加载的模型"""
    return {
        "loaded_models": list(model_cache.keys()),
        "count": len(model_cache)
    }


@app.delete("/models/{model_path:path}")
async def unload_model(model_path: str):
    """卸载模型"""
    if model_path in model_cache:
        del model_cache[model_path]
        logger.info(f"模型已卸载: {model_path}")
        return {"message": "模型已卸载", "model_path": model_path}
    else:
        raise HTTPException(status_code=404, detail="模型未找到")


@app.get("/drugs")
async def list_drugs():
    """获取所有药品列表"""
    try:
        loader = DataLoader()
        drug_ids = loader.get_unique_drugs()
        return {
            "drugs": drug_ids,
            "count": len(drug_ids)
        }
    except Exception as e:
        logger.error(f"获取药品列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hospitals")
async def list_hospitals():
    """获取所有医院列表"""
    try:
        loader = DataLoader()
        hospital_ids = loader.get_unique_hospitals()
        return {
            "hospitals": hospital_ids,
            "count": len(hospital_ids)
        }
    except Exception as e:
        logger.error(f"获取医院列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动 FastAPI 服务...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
