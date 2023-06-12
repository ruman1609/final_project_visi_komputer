import cv2
import os
import random
import pandas as pd
import shutil
from tqdm import tqdm

os.chdir("../../Dataset")

IMAGE_SIZE = (128, 128)

asli = pd.read_csv("1_asli/label.csv")

class_0 = asli.loc[asli["level"] == 0].to_dict("records")
class_1 = asli.loc[asli["level"] == 1].to_dict("records")
class_2 = asli.loc[asli["level"] == 2].to_dict("records")
class_3 = asli.loc[asli["level"] == 3].to_dict("records")
class_4 = asli.loc[asli["level"] == 4].to_dict("records")



def read_label_map(label_map_path):
    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").replace("\"", "").strip()

            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None

    return items



label_map = read_label_map("label_map.pbtxt")



def check_or_clear_folder(path, is_need_clear):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if is_need_clear:
            print(f"Delete {path}")
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"Done - Delete {path}\n\n")



parent = "2_0_1_splitted"
check_or_clear_folder(parent, is_need_clear=True)

class_dicts = [class_0, class_1, class_2, class_3, class_4]



def input_data(class_data, type_data):
    index = random.randint(0, len(class_data) - 1)
    filename = f"{class_data[index]['image']}.jpeg"
    im = cv2.imread(os.path.join("2_0_0_enhanced", filename))
    if isinstance(im, type(None)):
        print(f"{class_data[index]['image']}.jpeg NOT FOUND!")
        class_data.pop(index)
        return
    
    dest_area = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    source_area = im.shape[0] * im.shape[1]
    im = cv2.resize(im, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA if source_area > dest_area else cv2.INTER_CUBIC)

    level = list(label_map.keys())[list(label_map.values()).index(class_data[index]['level'])]
    path = os.path.join(parent, type_data, level)
    check_or_clear_folder(path, is_need_clear=False)
    cv2.imwrite(os.path.join(path, filename), im)

    class_data.pop(index)



for data in class_dicts:
    total = len(data)
    train_frac = .8
    test_val_frac = (1 - train_frac) / 2
    for i in tqdm(range(int(total * train_frac))):
        input_data(data, "train")

    for i in tqdm(range(int(total * test_val_frac))):
        input_data(data, "validation")

    for i in tqdm(range(int(total * test_val_frac))):
        input_data(data, "test")



print("\n\n\nCOMPLETE!!!")