FROM python:3.11-slim

WORKDIR /app

# 系统依赖（python-docx 需要 libmagic，faiss 需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 复制项目代码
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
