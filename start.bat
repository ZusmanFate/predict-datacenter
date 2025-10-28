@echo off
REM 药品销量预测系统 - Windows 启动脚本

echo ========================================
echo 药品销量预测系统
echo ========================================
echo.

REM 检查虚拟环境
if not exist "venv\" (
    echo [警告] 虚拟环境不存在，正在创建...
    python -m venv venv
    echo [完成] 虚拟环境创建成功
    echo.
)

REM 激活虚拟环境
echo [1/4] 激活虚拟环境...
call venv\Scripts\activate.bat

REM 安装依赖
echo.
echo [2/4] 检查依赖...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

REM 检查数据
echo.
echo [3/4] 检查示例数据...
if not exist "data\sales.db" (
    echo [提示] 未找到数据库，正在生成示例数据...
    python scripts\generate_sample_data.py
    if %errorlevel% neq 0 (
        echo [错误] 示例数据生成失败
        pause
        exit /b 1
    )
)

REM 启动 API 服务
echo.
echo [4/4] 启动 API 服务...
echo.
echo ========================================
echo API 服务已启动！
echo 访问地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo ========================================
echo.
echo 按 Ctrl+C 停止服务
echo.

uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

pause
