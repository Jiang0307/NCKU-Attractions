FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有應用程式檔案（包含 model.pth）
COPY . .

# 啟動命令
CMD ["python", "backend-PyTorch.py"]

