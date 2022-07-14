import pyrebase
import time
import numpy as np
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
    model = load_model(model_path_keras)
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")