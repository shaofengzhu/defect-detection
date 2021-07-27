import os
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


train_src_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/srcImg"
train_label_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/label"

# List all of the files under the folder
src_image_names = os.listdir(train_src_image_folder)
print(src_image_names)
print(len(src_image_names))

# let's check one of the image
src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"

src_image_full_path = os.path.join(train_src_image_folder, src_image_name)
print(f"src={src_image_full_path}")
label_image_full_path = os.path.join(train_label_image_folder, src_image_name)
print(f"label={label_image_full_path}")


# open image
src_image = Image.open(src_image_full_path)
label_image = Image.open(label_image_full_path)

# show the image
print("src_image")
src_image.show()
print("label_image")
label_image.show()

src_image_resized = src_image.resize((72, 92), Image.ANTIALIAS)
label_image_resized = label_image.resize((72,92), Image.ANTIALIAS)
src_image_resized.show()
label_image_resized.show()

# now convert it to Tensor
src_image_tensor = TF.to_tensor(src_image)
label_image_tensor = TF.to_tensor(label_image)

print(src_image_tensor)
# We will see that it's three dimension data
print(src_image_tensor.size())

# we only need to get the first element, which is two-dimension array.
print(src_image_tensor[0])

# Print the resized tensor
src_image_resized_tensor = TF.to_tensor(src_image_resized)
print(src_image_resized_tensor.size())

label_image_resized_tensor = TF.to_tensor(label_image_resized)
plt.imshow(label_image_resized_tensor[0], cmap="gray")
plt.show()
