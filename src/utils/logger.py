"""
日志工具模块
使用 loguru 提供统一的日志管理
"""
import sys
from pathlib import Path
from loguru import logger
import yaml


def setup_logger(config_path: str = "config/config.yaml"):
    """
    配置日志系统
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        log_config = config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', 
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
        rotation = log_config.get('rotation', '500 MB')
        retention = log_config.get('retention', '10 days')
        
        # 日志目录
        log_dir = Path(config.get('paths', {}).get('logs_dir', 'logs'))
    else:
        log_level = 'INFO'
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        rotation = '500 MB'
        retention = '10 days'
        log_dir = Path('logs')
    
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # 添加文件输出 - 所有日志
    logger.add(
        log_dir / "app.log",
        format=log_format,
        level=log_level,
        rotation=rotation,
        retention=retention,
        encoding='utf-8'
    )
    
    # 添加文件输出 - 错误日志
    logger.add(
        log_dir / "error.log",
        format=log_format,
        level="ERROR",
        rotation=rotation,
        retention=retention,
        encoding='utf-8'
    )
    
    logger.info(f"日志系统初始化完成 - Level: {log_level}")
    return logger


def get_logger(name: str = None):
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logger 实例
    """
    if name:
        return logger.bind(name=name)
    return logger


# 默认初始化
try:
    setup_logger()
except Exception as e:
    logger.warning(f"无法从配置文件加载日志配置: {e}，使用默认配置")
