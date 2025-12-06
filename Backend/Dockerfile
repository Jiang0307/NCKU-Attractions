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
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir grpcio==1.32.0 requests==2.11.1 numpy==1.19.5 && \
    pip install --no-cache-dir tensorflow==2.4.0 --no-deps && \
    pip install --no-cache-dir tensorflow-estimator==2.4.0 tensorflow-addons==0.19.0 silence-tensorflow==1.2.0 Pyrebase==3.0.27 Pillow==9.2.0 && \
    pip install --no-cache-dir absl-py astunparse flatbuffers google-pasta h5py keras-preprocessing opt-einsum protobuf==3.19.6 six termcolor typing-extensions wrapt gast && \
    pip cache purge && rm -rf /root/.cache/pip

# 複製應用程式檔案
COPY . .

# 創建必要的目錄
RUN mkdir -p data model

# 設置環境變數
ENV PYTHONUNBUFFERED=1

# 啟動應用程式
CMD ["python", "backend-Keras.py"]

