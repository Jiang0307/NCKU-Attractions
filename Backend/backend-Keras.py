import pyrebase
import time
import os
import cv2
import keras
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from keras.models import load_model
from pathlib import Path
# from datetime import time

# firebase設定檔
config = {"apiKey": "AIzaSyARu5SRV8FSTygZ5Q0uktmn-gb9vPRuB00","authDomain": "schoolspots-5a845.firebaseapp.com","databaseURL":"https://schoolspots-5a845-default-rtdb.asia-southeast1.firebasedatabase.app","projectId": "schoolspots-5a845","storageBucket": "schoolspots-5a845.appspot.com","messagingSenderId": "616225542838","appId": "1:616225542838:web:1e2d2c6167cfe78aac8c17"}
label_dict = {0:"博物館",1:"圖書館",2:"小西門",3:"招弟",4:"文物館",  5:"未來館",6:"榕園",7:"永恆之光",8:"直升機",9:"衛戍醫院",10:"詩人",11:"門神",12:"飛撲"}
dir_path = os.path.dirname(__file__)

def preprocess(img):
    img = cv2.resize(img,(224,224))
    test_data = np.expand_dims(img , axis=0)
    return test_data

def predict(test_data):
    result_array = model.predict(test_data)
    result = label_dict[np.argmax(result_array)]
    return result

def start_prediction():
    img = cv2.imread( Path(dir_path).joinpath("test_data").joinpath("test_data.jpg").as_posix() )
    test_data = preprocess(img)
    result = predict(test_data)
    return result

def stream_handler(message):
    cloud_path = "spots-1.jpg"
    if message["data"] == 1:
        print(f"\nProcessing...")
        begin = time.time()
        storage.child(cloud_path).download( Path(dir_path).joinpath("test_data").joinpath("test_data.jpg").as_posix() )
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
    model = load_model( Path(dir_path).joinpath("model").joinpath("model-Keras.h5").as_posix() )
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")