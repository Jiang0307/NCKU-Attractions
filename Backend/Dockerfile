# 使用 Python 3.10 slim 作為基礎映像（約 120MB）
FROM python:3.10-slim

# 設置工作目錄
WORKDIR /app

# 安裝最小系統依賴（僅 Pillow 需要的 libjpeg）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 複製 requirements.txt 並安裝 Python 依賴
# 使用 CPU-only 版本的 PyTorch 以大幅減少映像大小（從 ~7GB 減少到 ~2-3GB）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# 複製應用程式檔案
COPY . .

# 調試：列出所有檔案以確認複製成功
RUN echo "=== Files in /app ===" && \
    ls -la /app/ && \
    echo "=== Checking for backend-PyTorch.py ===" && \
    test -f /app/backend-PyTorch.py && echo "✓ backend-PyTorch.py found" || (echo "✗ backend-PyTorch.py NOT FOUND" && exit 1) && \
    echo "=== All .py files ===" && \
    find /app -name "*.py" -type f

# 創建必要的目錄
RUN mkdir -p data model

# 設置環境變數
ENV PYTHONUNBUFFERED=1

# 啟動應用程式
CMD ["python", "backend-PyTorch.py"]

