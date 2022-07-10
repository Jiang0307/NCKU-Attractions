import os,glob,shutil,tqdm

def check_dir(_dir):
    try:
        shutil.rmtree(_dir)
    except:
        pass
    finally:
        os.makedirs(_dir)

types = ["train","test"]
for _type in types:
    original_path = os.path.join(os.path.dirname(__file__),"converted_model.tflite")

    class_dir = rf"D:\code\peter\scene_rec\code\origin_dataset\{_type}"
    classes = glob.glob(os.path.join(class_dir,"*"))

    dataset_dir = r"D:\code\peter\scene_rec\code\dataset"
    _dataset_dir = os.path.join(dataset_dir,f"{_type}")
    img_dataset_dir = os.path.join(_dataset_dir,"img")
    check_dir(img_dataset_dir)

    # 輸出名稱和對應標籤
    label_map = {os.path.basename(c):i for i,c in enumerate(classes)}
    with open(os.path.join(dataset_dir,"classes_map.txt"),"w") as f:
        for cls_name,label in label_map.items():
            f.write(f"{cls_name}\t{label}\n")

    output_count = 0
    with open(os.path.join(_dataset_dir,"label.txt"),"w") as f:
        for i,class_full_path in enumerate(classes,1):
            img_paths = glob.glob(os.path.join(class_full_path,"*"))
            cls_name = os.path.basename(class_full_path)
            for img_path in tqdm.tqdm(img_paths,desc = f"{i}/{len(classes)}"):
                output_img_filename = os.path.join(img_dataset_dir,f"{output_count}.jpg")
                # 集中圖片
                shutil.copyfile(img_path,output_img_filename)

                cls_label = label_map[cls_name]

                # 只記檔名和label
                f.write(f"{output_count}.jpg\t{cls_label}\n")
                output_count += 1
