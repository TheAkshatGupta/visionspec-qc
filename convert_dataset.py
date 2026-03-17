import os
import shutil
import random
import xml.etree.ElementTree as ET

image_folder = "PCB_Dataset/JPEGImages"
xml_folder = "PCB_Dataset/Annotations"

train_pass = "dataset/train/pass"
train_defect = "dataset/train/defect"
val_pass = "dataset/validation/pass"
val_defect = "dataset/validation/defect"

os.makedirs(train_pass, exist_ok=True)
os.makedirs(train_defect, exist_ok=True)
os.makedirs(val_pass, exist_ok=True)
os.makedirs(val_defect, exist_ok=True)

images = os.listdir(image_folder)

pass_imgs = []
defect_imgs = []

for img in images:
    xml_file = img.replace(".jpg", ".xml")
    xml_path = os.path.join(xml_folder, xml_file)

    if not os.path.exists(xml_path):
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = root.findall("object")

    if len(objects) == 0:
        pass_imgs.append(img)
    else:
        defect_imgs.append(img)

random.shuffle(pass_imgs)
random.shuffle(defect_imgs)

split_pass = int(0.8 * len(pass_imgs))
split_defect = int(0.8 * len(defect_imgs))

train_pass_imgs = pass_imgs[:split_pass]
val_pass_imgs = pass_imgs[split_pass:]

train_defect_imgs = defect_imgs[:split_defect]
val_defect_imgs = defect_imgs[split_defect:]

for img in train_pass_imgs:
    shutil.copy(os.path.join(image_folder, img), train_pass)

for img in val_pass_imgs:
    shutil.copy(os.path.join(image_folder, img), val_pass)

for img in train_defect_imgs:
    shutil.copy(os.path.join(image_folder, img), train_defect)

for img in val_defect_imgs:
    shutil.copy(os.path.join(image_folder, img), val_defect)

print("Dataset ready ✅")