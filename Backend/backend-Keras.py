import pyrebase
import time
import shutil
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image
from config import *

def download_model_from_storage(storage, model_storage_path, local_path):
    """從 Firebase Storage 下載模型檔案（使用 pyrebase）"""
    try:
        print(f"正在從 Firebase Storage 下載模型: {model_storage_path}")
        # 確保目錄存在
        local_path_obj = Path(local_path)
        local_dir = local_path_obj.parent
        local_dir.mkdir(parents=True, exist_ok=True)
        filename = local_path_obj.name
        
        # 確保目標路徑不是目錄
        if local_path_obj.exists() and local_path_obj.is_dir():
            # 如果是目錄，刪除它
            shutil.rmtree(local_path_obj)
        
        # pyrebase 的 download() 需要 path（目錄）和 filename（檔名）兩個參數
        storage.child(model_storage_path).download(path=local_dir.as_posix(), filename=filename)
        
        # 驗證下載的檔案是否存在且是檔案
        if not local_path_obj.exists():
            print(f"錯誤：下載後檔案不存在: {local_path}")
            return False
        
        if local_path_obj.is_dir():
            print(f"錯誤：下載後路徑是目錄而非檔案: {local_path}")
            return False
        
        file_size = local_path_obj.stat().st_size
        if file_size == 0:
            print(f"錯誤：下載的檔案為空: {local_path}")
            return False
        
        print(f"模型下載完成: {local_path} (大小: {file_size} bytes)")
        return True
    except Exception as e:
        print(f"下載模型時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_model_from_file(model_path, storage=None):
    """載入模型，如果本地不存在則從 Firebase Storage 下載"""
    # 檢查本地模型檔案是否存在
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        if storage is None:
            raise FileNotFoundError(f"模型檔案不存在: {model_path}，且未提供 storage 物件")
        
        # 確保 model 目錄存在
        model_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 從 Firebase Storage 下載模型
        print("使用 Firebase Storage 下載模型...")
        if not download_model_from_storage(storage, model_storage_path_keras, model_path):
            raise FileNotFoundError(f"無法從 Firebase Storage 下載模型: {model_storage_path_keras}")
    
    # 驗證檔案是否存在且是檔案（不是目錄）
    if not model_path_obj.is_file():
        if model_path_obj.is_dir():
            raise ValueError(f"模型路徑指向目錄而非檔案: {model_path}")
        raise FileNotFoundError(f"模型檔案不存在: {model_path}")
    
    # 檢查檔案大小，確保不是空檔案
    if model_path_obj.stat().st_size == 0:
        raise ValueError(f"模型檔案為空: {model_path}")
    
    # 載入模型前，先驗證檔案格式
    print(f"正在載入模型: {model_path}")
    
    # 對於 .h5 檔案，先驗證是否為有效的 HDF5 格式
    if model_path.endswith('.h5'):
        try:
            # 檢查檔案開頭是否為 HDF5 格式的魔術數字
            with open(model_path, 'rb') as f:
                header = f.read(8)
                # HDF5 檔案開頭應該是 \x89HDF\r\n\x1a\n
                if header[:8] != b'\x89HDF\r\n\x1a\n':
                    raise ValueError(f"檔案 {model_path} 不是有效的 HDF5 格式（檔案頭: {header[:8]}）")
            print("檔案格式驗證通過（HDF5）")
        except Exception as format_check_error:
            print(f"檔案格式驗證失敗: {format_check_error}")
            # 如果格式不對，刪除檔案以便重新下載
            if model_path_obj.exists() and model_path_obj.is_file():
                try:
                    model_path_obj.unlink()
                    print(f"已刪除格式錯誤的檔案，將重新下載")
                except:
                    pass
            raise ValueError(f"模型檔案格式不正確: {format_check_error}")
    
    # 載入模型
    try:
        # 嘗試標準載入
        model = load_model(model_path)
        print("模型載入成功")
        return model
    except Exception as e:
        print(f"標準載入失敗: {e}")
        # 嘗試使用 compile=False 載入（可能解決某些版本兼容問題）
        try:
            print("嘗試使用 compile=False 載入模型...")
            model = load_model(model_path, compile=False)
            print("模型載入成功（使用 compile=False）")
            return model
        except Exception as e2:
            print(f"使用 compile=False 也失敗: {e2}")
            print("這可能是因為：")
            print("1. 模型檔案損壞或不完整")
            print("2. TensorFlow/Keras 版本不兼容")
            print("3. 模型架構問題")
            print("4. Firebase Storage 中的模型與本地模型不同")
            # 如果載入失敗，刪除檔案以便重新下載
            if model_path_obj.exists() and model_path_obj.is_file():
                try:
                    model_path_obj.unlink()
                    print(f"已刪除損壞的檔案，下次將重新下載")
                except:
                    pass
            raise e2

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

def stream_handler(message):
    cloud_path = "test.jpg"
    if message["data"] == 1:
        print(f"\nProcessing...")
        begin = time.time()
        
        # 確保 data 目錄存在
        local_dir = Path(dir_path).joinpath("data")
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # 下載圖片（pyrebase 需要 path 和 filename 兩個參數）
        img_file = local_dir.joinpath("test.jpg")
        print(f"正在從 Firebase Storage 下載圖片: {cloud_path} 到 {img_file}")
        storage.child(cloud_path).download(path=local_dir.as_posix(), filename="test.jpg")
        
        # 驗證檔案是否存在
        if not img_file.exists():
            print(f"錯誤：下載後檔案不存在: {img_file}")
            data = {"result":"", "pictureStatus":3}
            db.update(data)
            return
        
        if img_file.is_dir():
            print(f"錯誤：下載後路徑是目錄而非檔案: {img_file}")
            data = {"result":"", "pictureStatus":3}
            db.update(data)
            return
        
        file_size = img_file.stat().st_size
        if file_size == 0:
            print(f"錯誤：下載的圖片檔案為空: {img_file}")
            data = {"result":"", "pictureStatus":3}
            db.update(data)
            return
        
        print(f"圖片下載完成: {img_file} (大小: {file_size} bytes)")
        
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
    # 統一初始化 Firebase（只初始化一次）
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    
    # 載入模型
    print("正在載入模型...")
    
    # 先檢查本地是否有模型檔案
    model_path_obj = Path(model_path_keras)
    if model_path_obj.exists() and model_path_obj.is_file():
        print("找到本地模型檔案，直接載入")
        try:
            model = load_model_from_file(model_path_keras)
        except Exception as e:
            print(f"載入本地模型失敗: {e}，將重新下載")
            # 刪除損壞的檔案
            try:
                if model_path_obj.exists():
                    model_path_obj.unlink()
            except:
                pass
            # 從 Firebase Storage 下載
            model = load_model_from_file(
                model_path_keras, 
                storage=storage
            )
    else:
        # 如果沒有，從 Firebase Storage 下載
        print("本地模型不存在，開始下載...")
        model = load_model_from_file(
            model_path_keras, 
            storage=storage
        )
    
    print("模型載入完成")
    
    # 開始監聽 Firebase Realtime Database
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")