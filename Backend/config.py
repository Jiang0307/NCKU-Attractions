import os
from pathlib import Path

# firebase設定檔
config = {"apiKey":"AIzaSyARu5SRV8FSTygZ5Q0uktmn-gb9vPRuB00" , "authDomain":"schoolspots-5a845.firebaseapp.com" , "databaseURL":"https://schoolspots-5a845-default-rtdb.asia-southeast1.firebasedatabase.app" , "projectId":"schoolspots-5a845" , "storageBucket":"schoolspots-5a845.appspot.com" , "messagingSenderId":"616225542838" , "appId":"1:616225542838:web:1e2d2c6167cfe78aac8c17"}
label_dict = {0:"午後的心願",1:"博物館",2:"圖書館",3:"小西門",4:"思想者",5:"思量",6:"成功湖",7:"招弟",8:"文物館",9:"未來館",10:"校長官舍",11:"格致堂",12:"榕園",13:"歷史系館",14:"永恆之光",15:"浮雲樹影",16:"直升機",17:"衛戍醫院",18:"詩人",19:"資訊系館",20:"門神",21:"雨豆樹",22:"飛撲",}
dir_path = os.path.dirname(__file__)
img_path = Path(dir_path).joinpath("test_data").joinpath("test_data.jpg").as_posix()
model_path_keras = Path(dir_path).joinpath("model").joinpath("model-Keras.h5").as_posix()
model_path_pytorch = Path(dir_path).joinpath("model").joinpath("model-PyTorch.pth").as_posix()