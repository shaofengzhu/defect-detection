import os
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")

src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"
src_image = Image.open(os.path.join(train_src_image_folder, src_image_name))

src_image_data = np.array(src_image, dtype=np.float32)

src_image_tensor = torch.tensor(src_image_data)
print(src_image_tensor.shape)

# The data is 2 dimensional, convert it to 4 dimensional. It's because nn.Conv2d requires 4 dimensional data. 
# the first dimensional is the the index in the batch
# the second dimensional is channel
# the third and fourth is the 2D data
src_image_tensor = torch.unsqueeze(src_image_tensor, dim=0)
src_image_tensor = torch.unsqueeze(src_image_tensor, dim=0)
print(src_image_tensor.shape)


conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_result = conv.forward(src_image_tensor)
print(conv_result.shape)

conv_result_data = conv_result.squeeze().detach().numpy()

fig, axarr = plt.subplots(2)
# the first is source image
axarr[0].imshow(src_image_data)
# the second is the image after convolutional process
axarr[1].imshow(conv_result_data)
plt.show()
