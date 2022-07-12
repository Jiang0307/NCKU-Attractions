import os
import gc
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms , datasets , models
from torchsummary import summary
from pathlib import Path
from PIL import Image
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm
gc.collect()
torch.cuda.empty_cache()

DATA_PATH = Path("C:/Users/User/Desktop/DATA")
CURRENT_PATH = os.path.dirname(__file__)
DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
EPOCH = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WORKERS = 0
CRITERION = nn.CrossEntropyLoss()

def count_class(): 
    count = 0
    folder_1 = os.scandir(DATA_PATH)
    for sites in folder_1:
        if sites.is_dir():
            count += 1
    print(f"class count : {count}\n")
    return count

def load_data():
    print(f"load_data\n")
    train_preprocess = transforms.Compose([transforms.Resize((224,224)) , transforms.ToTensor()])
    test_preprocess = transforms.Compose([transforms.Resize((224,224)) , transforms.ToTensor()])

    train_data  = datasets.ImageFolder( Path(CURRENT_PATH).joinpath("Data").joinpath("train") , transform=train_preprocess)
    test_data  = datasets.ImageFolder( Path(CURRENT_PATH).joinpath("Data").joinpath("test") , transform=test_preprocess)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=WORKERS,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,  num_workers=WORKERS)
    print(train_data.class_to_idx)
    return train_loader , validation_loader

def build_model():
    print(f"build_model\n")
    num_class = count_class()
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc0 = nn.Linear(num_ftrs, 256)
    model.fc1 = nn.Linear(256 , 128)
    model.fc2 = nn.Linear(128 , 64)
    model.fc3 = nn.Linear(64 , num_class)
    model.to(DEVICE)
    return model

def train(train_loader , validation_loader):
    print(f"train\n")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    valid_loss_min = np.Inf
    for epoch in range(1, EPOCH+1):
        train_loss = 0.0
        valid_loss = 0.0
        print("running epoch: {}".format(epoch))
        # train
        model.train()
        for data , target in tqdm(train_loader):
            if DEVICE == "cuda":
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = CRITERION(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            del loss , output
        # validation
        model.eval()
        for data, target in tqdm(validation_loader):
            # move tensors to GPU if CUDA is available
            if DEVICE == "cuda":
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = CRITERION(output, target)
            valid_loss += loss.item()*data.size(0)
            del loss , output

        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(validation_loader.dataset)
            
        print("\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            save_path = Path(os.path.dirname(__file__)).joinpath("model-PyTorch.pth").as_posix()
            print("Validation loss decreased ({:.6f} --> {:.6f}).".format(valid_loss_min,valid_loss))
            print(f"Save model to {save_path}")
            torch.save(model , save_path )
            valid_loss_min = valid_loss

if __name__ == "__main__":
    print(f"Device : {DEVICE}\n")
    train_loader , validation_loader = load_data()
    model = build_model()
    # train(train_loader , validation_loader)