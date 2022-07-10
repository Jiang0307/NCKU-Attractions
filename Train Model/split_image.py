import os,glob,shutil
import random,tqdm
from cv2 import cv2
from PIL import Image
import numpy as np
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    IAAPiecewiseAffine,
    GaussianBlur,
    Blur,
    CoarseDropout,
    GaussNoise,
    ToGray,
    OneOf,
    HueSaturationValue,
    RGBShift,
    CLAHE,
    ChannelShuffle,
    InvertImg,
    RandomFog,
    RandomBrightness,
    ChannelDropout,
    RandomResizedCrop,
    ShiftScaleRotate
)

def transform(img_shape):
    # 圖片前處理
    h,w = img_shape[:2]
    return Compose(
        [
            ShiftScaleRotate(),
            CoarseDropout(75,4,4),
            OneOf(
                [
                    GaussianBlur(blur_limit = (3,5)),
                    Blur(blur_limit = (3,5))
                ]
            ),
            GaussNoise(),
            OneOf([
                ChannelDropout(),
                OneOf([
                    HueSaturationValue(),
                    RGBShift(),
                    ChannelShuffle(),
                    InvertImg(),
                    ToGray(),
                ])
            ]),
            RandomBrightness(),
            RandomFog(),
            HorizontalFlip(),
            VerticalFlip(),
            IAAPiecewiseAffine((0.01,0.02)),
            RandomResizedCrop(h,w,(0.85,1))
        ]
    )

def check_dir(_dir):
    # 建立目錄
    try:
        shutil.rmtree(_dir)
    except:
        pass
    finally:
        os.makedirs(_dir)

split_ratio = 0.8
img_shape = (224,224)

img_dataset_path = r'D:\code\peter\scene_rec\code\專題手動蒐集'
img_origin_dataset_path = r'D:\code\peter\scene_rec\code\origin_dataset'

cls_dirs = glob.glob(os.path.join(img_dataset_path,'*'))
for i,img_dir in enumerate(cls_dirs,1):
    # 得到所有目錄
    classname = os.path.basename(img_dir)

    # 搜遍一個目錄下所有圖片
    imgs = glob.glob(os.path.join(img_dir,'*'))
    random.shuffle(imgs)

    # 把圖片分成訓練和測試
    train_img_paths = imgs[:int(len(imgs) * split_ratio)]
    test_img_paths = imgs[int(len(imgs) * split_ratio):]

    train_class_dir = os.path.join(img_origin_dataset_path,'train',classname)
    check_dir(train_class_dir)
    for img_path in tqdm.tqdm(train_img_paths,f'train set: {i}/{len(cls_dirs)}',maxinterval = 0.5):
        img_basename = os.path.basename(img_path)

        # 因為圖片數量差距大，所以對數量少的圖，每一張都多做幾次增強
        aug_per_img = 1
        if len(train_img_paths) < 100:
            while True:
                num = len(train_img_paths) * aug_per_img
                if 80 < num :
                    break
                else:
                    aug_per_img += 1

        # 前處理
        try:
            img = Image.open(img_path)
        except:
            # 如果圖片毀損就跳過
            continue
        img = np.array(img,np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for n in range(aug_per_img):
            tmp = img.copy()
            # 把圖片全縮小成目標尺寸(224, 224)
            tmp = cv2.resize(tmp,img_shape)
            aug = transform(tmp.shape)
            tmp = aug(image = tmp)['image']
            tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

            output_path = os.path.join(train_class_dir,img_basename[:-4] + f'_{n}' + img_basename[-4:])
            tmp = Image.fromarray(tmp)
            tmp.save(output_path)
    
    test_class_dir = os.path.join(img_origin_dataset_path,'test',classname)
    check_dir(test_class_dir)
    for img_path in tqdm.tqdm(test_img_paths,f'test set: {i}/{len(cls_dirs)}',maxinterval = 0.5):
        img_basename = os.path.basename(img_path)
        # 前處理
        try:
            img = Image.open(img_path)
        except:
            # 如果圖片毀損就跳過
            continue

        img = np.array(img,np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,img_shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        output_path = os.path.join(test_class_dir,img_basename[:-4] + f'_{n}' + img_basename[-4:])
        img = Image.fromarray(img)
        img.save(output_path)

    # break