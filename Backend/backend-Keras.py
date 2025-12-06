import pyrebase
import time
import numpy as np
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image
from config import *

def preprocess():
    PIL_img = Image.open(img_path)
    resized_img = PIL_img.resize((224, 224))
    img = np.asarray(resized_img)
    test_data = np.expand_dims(img , axis=0)
    return test_data

def predict(test_data):
    result_array = model.predict(test_data)
    result = label_dict[np.argmax(result_array)]
    return result

def start_prediction():
    test_data = preprocess()
    result = predict(test_data)
    return result

def download_model_from_storage(storage, model_storage_path, local_path):
    """從 Firebase Storage 下載模型檔案"""
    try:
        print(f"正在從 Firebase Storage 下載模型: {model_storage_path}")
        # 確保目錄存在
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)
        # 使用舊版本的簡單下載方式（只傳一個參數：完整路徑）
        storage.child(model_storage_path).download(local_path)
        print(f"模型下載完成: {local_path}")
        return True
    except Exception as e:
        print(f"下載模型時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

class HealthCheckHandler(BaseHTTPRequestHandler):
    """簡單的健康檢查 HTTP 處理器（用於 Cloud Run）"""
    def do_GET(self):
        if self.path == '/health' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "service": "ncku-attractions-backend"}')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # 抑制 HTTP 服務器的日誌輸出
        pass

def start_http_server(port=8080):
    """在後台線程中啟動 HTTP 服務器（用於 Cloud Run 健康檢查）"""
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"HTTP 健康檢查服務器已啟動，監聽端口 {port}")
    server.serve_forever()

def stream_handler(message):
    cloud_path = "test.jpg"
    if message["data"] == 1:
        print(f"\nProcessing...")
        begin = time.time()
        storage.child(cloud_path).download( Path(dir_path).joinpath("data").joinpath("test.jpg").as_posix() )
        result = start_prediction() # 進行影像辨識處理區段，把顯示結果填到result
        if result != "": # 有辨識結果為2
            print(f"Result : {result}")
            data = {"pictureStatus":2}
            data["result"] = result
            db.update(data)
        else: # 無法辨識結果為3
            print(f"Result : Unknown")
            data = {"result":"", "pictureStatus":3}
            db.update(data)
        end = time.time()
        print(f"Time Elapsed : {round(end-begin,2)}s")

if __name__ == "__main__":
    # 初始化 Firebase
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    
    # 載入模型（如果本地不存在則從 Firebase Storage 下載）
    print("正在載入模型...")
    model_path_obj = Path(model_path_keras)
    
    if not model_path_obj.exists():
        # 本地模型不存在，從 Firebase Storage 下載
        print("本地模型不存在，開始從 Firebase Storage 下載...")
        if not download_model_from_storage(storage, model_storage_path_keras, model_path_keras):
            raise FileNotFoundError(f"無法從 Firebase Storage 下載模型: {model_storage_path_keras}")
    else:
        print("找到本地模型檔案，直接載入")
    
    # 載入模型
    model = load_model(model_path_keras)
    print("模型載入完成")
    
    # 啟動 HTTP 服務器（用於 Railway/Cloud Run 健康檢查）
    port = int(os.environ.get('PORT', 8080))
    http_thread = threading.Thread(target=start_http_server, args=(port,), daemon=True)
    http_thread.start()
    print(f"HTTP 服務器已在後台線程啟動，端口: {port}")
    
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")
