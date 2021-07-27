import os
import torch
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn


train_src_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/srcImg"
train_label_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/label"

src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"

src_image_full_path = os.path.join(train_src_image_folder, src_image_name)
print(f"src={src_image_full_path}")
label_image_full_path = os.path.join(train_label_image_folder, src_image_name)
print(f"label={label_image_full_path}")


src_image = Image.open(src_image_full_path)
print(src_image.size)


unique_values = []
label_image = Image.open(label_image_full_path)
px = label_image.load()
print(px)
for x in range(label_image.size[0]):
    for y in range(label_image.size[1]):
        v = px[x,y]
        if v not in unique_values:
            unique_values.append(v)

print(unique_values)


