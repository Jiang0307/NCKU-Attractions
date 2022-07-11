import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tensorflow.keras.layers import Dense , GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50 , InceptionV3 , InceptionResNetV2 , Xception , MobileNetV2	
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import EarlyStopping

base_path = "C:/Users/User/Desktop/DATA"
BATCH_SIZE = 32

selected_model = "ResNet50"
model_dict={"ResNet50":0,"MobileNetV2":1,"InceptionResNetV2":2,"Xception":3}
model_input_size={"ResNet50":(224,224),"MobileNetV2":(224,224),"InceptionResNetV2":(299,299),"Xception":(299,299)}
model_input_shape={"ResNet50":(224,224,3),"MobileNetV2":(224,224,3),"InceptionResNetV2":(299,299,3),"Xception":(299,299,3)}

site_list = [] #存放所有景點的list，有新資料夾的話會自動append
site_name_dict = {} #依照景點的字典，裡面存完整路徑&檔名
img_label_dict = {} #依照景點的字典，各景點對應的label

#計算資料夾數(景點數量)
def count_class(): 
    count = 0
    folder_1 = os.scandir(base_path)
    for sites in folder_1:
        if sites.is_dir():
            count += 1
            site_list.append(sites.name) #因為會慢慢加入景點，所以用讀取資料夾名稱來append到list中，避免手刻site_list
    return count

#讀檔，把圖片都放進train跟test的list中，count為景點數
def preprocess(count,train,test): 
    list_of_sites_name = [[] for i in range(count)] #分配出count個list
    for i in range(count):
        name = site_list[i]
        site_name_dict[name] = list_of_sites_name[i]
        img_label_dict[name] = i #標記每個景點的label是哪個數字
        print(f"i : {i} , name : {name}")

    folder_1 = os.scandir(base_path)
    for folder_2 in folder_1: #folder2 : 各景點的資料夾
        if folder_2.is_dir(): 
            print(folder_2.name)
            temp_path = os.path.join(base_path,folder_2) #temp_path : 各景點資料夾的路徑
            category = os.path.basename(temp_path) #景點的資料夾名稱
            for file in os.listdir(folder_2):            
                if file.endswith(".jpg"):
                    full_path=os.path.join(temp_path,file) #full_path : 各景點資料夾中每張照片的完整路徑
                    site_name_dict[category].append(full_path)

    for category in site_list:
        temp = []
        folder_size = len(site_name_dict[category])
        for img in site_name_dict[category]:
            try:
                im = Image.open(img)
                if im is not None:
                    im = im.resize(model_input_size[selected_model] , Image.ANTIALIAS)
                    imarray = np.asarray(im,dtype="int32")
                    if( imarray.shape == model_input_shape[selected_model] ): #這邊這樣做是因為有些圖片是4個色彩通道，無法當作input所以不管4通道的圖片
                        #imarray = np.asarray(im,dtype="int32")
                        label = img_label_dict[category]
                        temp.append([im,label]) #資料結構是[image,label]包在一起
            except(OSError,NameError):
                None
        length = int(folder_size*0.2) #每個景點分配20%的照片當test data
        train += temp[length:]
        test += temp[:length]
    
    print(f"train : {len(train)}")
    print(f"test : {len(test)}")

def data_augmentation(train , augmentation):
    for i in range(len(train)):
        img = train[i][0] #TRAIN[i][1]是label
        label = train[i][1]
        #資料增強的部分顯卡內存不夠所以有些註解掉
        # brighter
        value1 = random.randint(0,255)
        temp = np.asarray(img)
        shape = temp.shape
        mat = np.ones(shape , dtype = "int32") * value1
        brighter = cv2.add(np.asarray(img,dtype="int32") , mat) 
        brighter = np.asarray(brighter,dtype="int32")
        augmentation.append([brighter, label])
        # darker
        value2 = random.randint(0,255)
        temp2 = np.asarray(img)
        shape2 = temp2.shape
        mat2 = np.ones(shape2 , dtype = "int32") * value2
        darker = cv2.subtract(np.asarray(img,dtype="int32") , mat2) 
        darker = np.asarray(darker,dtype="int32")
        augmentation.append([darker, label])
        # counter-clockwise rotation
        counter_clockwise = img.rotate(random.randint(0,30))
        counter_clockwise = np.asarray(counter_clockwise,dtype="int32")
        augmentation.append([counter_clockwise, label])
        # clockwise rotation
        clockwise = img.rotate(-1 * random.randint(0,30))
        clockwise = np.asarray(clockwise,dtype="int32")
        augmentation.append([clockwise, label])
        # horizontal flip
        h_flip = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        h_flip = np.asarray(h_flip,dtype="int32")
        augmentation.append([h_flip , label])
        """
        #central crop
        central_crop = tf.image.central_crop(img,0.5)
        central_crop = array_to_img(central_crop)
        central_crop = np.asarray(central_crop,dtype="int32")
        augmentation.append([central_crop, label])
        # hue
        hue = tf.image.adjust_hue(img, (random.randint(2,5)/10))
        hue = np.asarray(hue,dtype="int32")
        augmentation.append([hue, label])
        """

#用keras的pre-trained model
def build_model(model_name,n_classes): 
    if model_name == "ResNet50":
        selected = ResNet50(include_top = False , weights = "imagenet" , input_shape = model_input_shape[model_name])
    elif model_name == "MobileNetV2":
        selected = MobileNetV2(include_top = False , weights = "imagenet" , input_shape = model_input_shape[model_name])
    elif model_name == "InceptionResNetV2":
        selected = InceptionResNetV2(include_top = False , weights="imagenet" , input_shape = model_input_shape[model_name])
    elif model_name == "Xception":
        selected = Xception(include_top = False , weights="imagenet" , input_shape = model_input_shape[model_name])
    model = Sequential()
    model.add(selected)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))
    selected_optimizer = Adam(learning_rate=0.00001)
    model.compile(optimizer=selected_optimizer , loss="sparse_categorical_crossentropy" , metrics=["accuracy"])
    model.summary()
    return model

def create_dataset(train , test , augmentation):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    #圖片轉陣列
    for i in range(len(train)): 
        temp = train[i][0] 
        temp = np.asarray(temp,dtype="int32") 
        train[i][0] = temp
    #圖片轉陣列
    for i in range(len(test)):
        temp = test[i][0]
        temp = np.asarray(temp,dtype="int32")
        test[i][0] = temp

    train_temp = np.asarray(train)
    test_temp = np.asarray(test)
    augmentation_temp = np.asarray(augmentation)
    print(train_temp.shape, augmentation_temp.shape)
    train_temp = np.concatenate((train_temp , augmentation_temp)) #把資料增強的圖片加到train_temp中
    print(train_temp.shape)
    np.random.shuffle(train_temp)
    #把image跟label分開
    for image, label in train_temp:
        train_data.append(image)
        train_label.append(label)
    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    #把image跟label分開
    for image, label in test_temp:
        test_data.append(image)
        test_label.append(label)
    test_data = np.asarray(test_data)
    test_label = np.asarray(test_label)
    
    return train_data , train_label , test_data , test_label

#回傳label對應的景點名稱
def get_key(predict_label , img_label_dict):
    new_dict = {}
    for key, value in img_label_dict.items():
        new_key = int(value)
        new_value = str(key)
        new_dict[new_key] = new_value
    predict_name = new_dict[predict_label]
    return predict_name

#印出accuracy跟loss
def plot_training_history(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"], loc="lower right")
    plt.show()
    # loss graph
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()

#用testing set測試後算準確率跟把預測錯的圖片印出來
def prediction_result(test_data,test_label,prediction,img_label_dict):
    correct = 0
    wrong_index = []
    predicted_label = []
    actual_label = []

    for i in range(len(prediction)):
        predict = np.argmax(prediction[i])
        actual = test_label[i]
        if (predict == actual):
            correct += 1
        else:
            wrong_index.append(i)
            actual_label.append(actual)
            predicted_label.append(predict)

    print("accuracy : ", correct / len(test_data) * 100)
    #print(wrong_index)
    #print(predicted_label)
    axes = []
    cols = 8
    rows = int(len(wrong_index) / cols) + 1
    figs = plt.figure(figsize=(80,80))

    for i in range(rows * cols):
        if(i < len(wrong_index)):
            axes.append(figs.add_subplot(rows, cols, i+1))
            img = array_to_img(test_data[wrong_index[i]])
            wrong = get_key(predicted_label[i],img_label_dict)
            answer = get_key(actual_label[i],img_label_dict)
            name = "答案 : "+answer+" 預測 : "+wrong
            plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"] 
            plt.rcParams["axes.unicode_minus"] = False
            axes[-1].set_title(name)
            plt.imshow(img)

train = []
test = []
augmentation = []

NUM_CLASS = count_class()
preprocess(NUM_CLASS , train , test)
data_augmentation(train , augmentation)
train_data , train_label  , test_data , test_label = create_dataset(train , test , augmentation)
print("train : " , len(train_data) , len(train_label))
print("test : " , len(test_data) , len(test_label))

model = build_model(selected_model , NUM_CLASS)
callback = EarlyStopping(monitor="val_accuracy", mode="max" , verbose=1 , restore_best_weights=True , patience=20)
history = model.fit(train_data , train_label , batch_size=BATCH_SIZE , epochs=100 , shuffle=True , validation_split=0.2 , callbacks=callback)
model.save(Path(os.path.dirname(__file__)).joinpath("model.h5").as_posix())  
plot_training_history(history)
prediction = model.predict(test_data)
prediction_result(test_data , test_label , prediction , img_label_dict)