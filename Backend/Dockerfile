# 使用 Python 3.10 slim 作為基礎映像
FROM python:3.10-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式檔案
COPY . .

# 創建必要的目錄
RUN mkdir -p data model

# 設置環境變數
ENV PYTHONUNBUFFERED=1

# 啟動應用程式
CMD ["python", "backend-PyTorch.py"]

