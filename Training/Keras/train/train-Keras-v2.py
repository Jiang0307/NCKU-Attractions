import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tensorflow.keras.layers import Dense , GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

CURRENT_PATH = os.path.dirname(__file__)
DATA_PATH = Path("C:/Users/User/Desktop/DATA")
TRAIN_PATH = Path(CURRENT_PATH).joinpath("Data").joinpath("train").as_posix()
TEST_PATH = Path(CURRENT_PATH).joinpath("Data").joinpath("test").as_posix()
EPOCH = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WORKERS = 0
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224,3)

def count_class(): 
    count = 0
    folder_1 = os.scandir(DATA_PATH)
    for sites in folder_1:
        if sites.is_dir():
            count += 1
    print(f"class count : {count}\n")
    return count

def build_model(num_classes):
    backbone = ResNet50(include_top=False , weights="imagenet" , input_shape=IMAGE_SHAPE)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    selected_optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=selected_optimizer , loss="categorical_crossentropy" , metrics=["accuracy"])
    model.summary()
    return model

def load_data():
    train_datagen = ImageDataGenerator(horizontal_flip=True)
    train_batches = train_datagen.flow_from_directory(TRAIN_PATH,target_size=IMAGE_SIZE,interpolation="bicubic",class_mode="categorical",shuffle=True,batch_size=BATCH_SIZE)   
    valid_datagen = ImageDataGenerator()
    validation_batches = valid_datagen.flow_from_directory(TEST_PATH,target_size=IMAGE_SIZE,interpolation="bicubic",class_mode="categorical",shuffle=False,batch_size=BATCH_SIZE)
    print("\nLabel : ")
    for cls, idx in train_batches.class_indices.items():
        print(f"{idx} = {cls}")
    
    return train_batches , validation_batches

def train(model , train_batches , validation_batches):
    model.fit_generator(train_batches , steps_per_epoch=train_batches.samples//BATCH_SIZE , validation_data=validation_batches , validation_steps=validation_batches.samples//BATCH_SIZE , epochs=EPOCH)
    model.save( Path(CURRENT_PATH).joinpath("model-Keras.h5").as_posix() )

if __name__ == "__main__":
    num_classes = count_class()
    model = build_model(num_classes)
    train_batches , validation_batches = load_data()
    train(model , train_batches , validation_batches)