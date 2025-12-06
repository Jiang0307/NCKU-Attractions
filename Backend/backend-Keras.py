import pyrebase
import time
import os
import numpy as np
import requests
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

# --------------------------
# 🔥 改成 requests 下載模型
# --------------------------
def download_model_via_requests(storage, firebase_path, local_path):
    try:
        print(f"準備下載模型: {firebase_path}")

        # 產生可下載的網址（用相對路徑）
        url = storage.child(firebase_path).get_url(None)
        print("模型下載 URL:", url)

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # 開始下載
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        size = os.path.getsize(local_path)
        print(f"模型下載完成，檔案大小: {size} bytes")

        if size < 1000:
            raise ValueError("模型檔案異常（可能 0 bytes），請檢查 Firebase Storage 權限")

        return True
    except Exception as e:
        print(f"下載模型失敗: {e}")
        return False

# --------------------------
# test.jpg 仍然可用 pyrebase4 或 requests
# --------------------------
def download_image(storage, firebase_path, local_path):
    """下載 test.jpg（小檔案）"""
    try:
        url = storage.child(firebase_path).get_url(None)
        r = requests.get(url)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        open(local_path, "wb").write(r.content)
        print("圖片下載完成")
    except Exception as e:
        print(f"下載圖片失敗: {e}")

def stream_handler(message):
    cloud_path = "test.jpg"
    if message["data"] == 1:
        print(f"\nProcessing...")
        begin = time.time()

        local_img_path = Path(dir_path).joinpath("data").joinpath("test.jpg").as_posix()
        download_image(storage, cloud_path, local_img_path)

        result = start_prediction()
        if result != "":
            print(f"Result : {result}")
            data = {"pictureStatus":2, "result": result}
            db.update(data)
        else:
            print(f"Result : Unknown")
            data = {"result":"", "pictureStatus":3}
            db.update(data)

        end = time.time()
        print(f"Time Elapsed : {round(end-begin,2)}s")

if __name__ == "__main__":
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()

    print("正在載入模型...")
    model_path_obj = Path(model_path_keras)

    # 🔥 若本地模型不存在 → 用 requests 下載
    if not model_path_obj.exists():
        print("本地模型不存在，開始下載模型...")
        ok = download_model_via_requests(storage, model_storage_path_keras, model_path_keras)
        if not ok:
            raise FileNotFoundError("模型下載失敗，無法繼續執行")
    else:
        print("找到本地模型檔案")

    # 檢查模型是否存在
    model_path_abs = os.path.abspath(model_path_keras)
    print(f"模型路徑: {model_path_abs}")
    print(f"模型大小: {os.path.getsize(model_path_abs)} bytes")

    # 載入模型
    try:
        model = load_model(model_path_abs, compile=False)
        print("模型載入完成")
    except Exception as e:
        print("模型載入失敗:", e)
        raise

    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")