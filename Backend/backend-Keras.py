import pyrebase
import time
import os
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image

# firebase設定檔
config = {"apiKey":"AIzaSyARu5SRV8FSTygZ5Q0uktmn-gb9vPRuB00" , "authDomain":"schoolspots-5a845.firebaseapp.com" , "databaseURL":"https://schoolspots-5a845-default-rtdb.asia-southeast1.firebasedatabase.app" , "projectId":"schoolspots-5a845" , "storageBucket":"schoolspots-5a845.appspot.com" , "messagingSenderId":"616225542838" , "appId":"1:616225542838:web:1e2d2c6167cfe78aac8c17"}
label_dict = {0:"午後的心願",1:"博物館",2:"圖書館",3:"小西門",4:"思想者",5:"思量",6:"成功湖",7:"招弟",8:"文物館",9:"未來館",10:"校長官舍",11:"格致堂",12:"榕園",13:"歷史系館",14:"永恆之光",15:"浮雲樹影",16:"直升機",17:"衛戍醫院",18:"詩人",19:"資訊系館",20:"門神",21:"雨豆樹",22:"飛撲",}
dir_path = os.path.dirname(__file__)
model_path = Path(dir_path).joinpath("model").joinpath("model-Keras.h5").as_posix()
img_path = Path(dir_path).joinpath("test_data").joinpath("test_data.jpg").as_posix()

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
    model = load_model(model_path)
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")