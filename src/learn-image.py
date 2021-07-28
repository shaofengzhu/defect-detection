from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# assume the following directory structure
# src/a.py
# dataset/train_val_image_label/train_image_label/srcImg/*.bmp
# dataset/train_val_image_label/train_image_label/label/*.bmp

# please change path if the directory is not same as the above.


current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")

src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"

src_image_full_path = os.path.join(train_src_image_folder, src_image_name)
print(f"src={src_image_full_path}")
label_image_full_path = os.path.join(train_label_image_folder, src_image_name)
print(f"label={label_image_full_path}")

# open image
src_image = Image.open(src_image_full_path)
print(src_image.size)
label_image = Image.open(label_image_full_path)
print(label_image.size)

# get image data into np.array
src_image_data = np.array(src_image, dtype=np.float32)
print(f"src_image_data.shape={src_image_data.shape}")
print(src_image_data.max())
# convert to 0 to 1
src_image_data = src_image_data / 255.0

label_image_data = np.array(label_image, dtype=np.float32)
print(label_image_data.max())
# replace all element greater than 0 with value 1.0
# we only want to use 0 or 1
label_image_data[label_image_data > 0] = 1.0
print(label_image_data.max())

# show two images
fig, axarr = plt.subplots(2)
# show the first image with src_image_data
axarr[0].imshow(src_image_data)
# show the second image with label_image_data
axarr[1].imshow(label_image_data)
plt.show()

