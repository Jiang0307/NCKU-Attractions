# 使用 Python 3.8 slim 作為基礎映像（與本地環境一致）
FROM python:3.8-slim

# 設置工作目錄
WORKDIR /app

# 安裝最小系統依賴（僅 Pillow 需要的 libjpeg）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 複製 requirements.txt 並安裝 Python 依賴
# 使用 TensorFlow/Keras（比 PyTorch 更輕量）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# 複製應用程式檔案
COPY . .

# 創建必要的目錄
RUN mkdir -p data model

# 設置環境變數
ENV PYTHONUNBUFFERED=1

# 啟動應用程式
CMD ["python", "backend-Keras.py"]

