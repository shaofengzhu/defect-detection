import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
test_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/srcImg")
test_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/val_label")

batch_size = 2
image_width = 716 // 4
image_height = 920 // 4
learning_rate = 0.001
num_epochs = 1

class ImageDataset(Dataset):
    def __init__(self, src_image_folder, label_image_folder, transform = None):
        super(ImageDataset, self).__init__()
        self.src_image_folder = src_image_folder
        self.label_image_folder = label_image_folder
        self.transform = transform
        self.src_img_names = os.listdir(src_image_folder)

    def __len__(self):
        return len(self.src_img_names)

    def __getitem__(self, index):
        src_img_loc = os.path.join(self.src_image_folder, self.src_img_names[index])
        src_image = Image.open(src_img_loc)
        # src_image = src_image.resize((image_width, image_height), Image.NEAREST)
        src_image_data = np.array(src_image, dtype=np.float32)
        src_image_data = src_image_data / 255.0
        # src_image_data = torch.tensor(src_image_data, dtype=torch.float32)


        label_img_loc = os.path.join(self.label_image_folder, self.src_img_names[index])
        label_image = Image.open(label_img_loc)
        # label_image = label_image.resize((image_width, image_height), Image.NEAREST)
        label_image_data = np.array(label_image, dtype=np.float32)
        label_image_data[label_image_data > 0] = 1.0
        # label_image_data = torch.tensor(label_image_data, dtype=torch.float32)

        if self.transform is not None:
            augmentations = self.transform(image=src_image_data, mask=label_image_data)
            src_image_data = augmentations["image"]
            label_image_data = augmentations["mask"]

        src_image_data_threechannel = torch.cat((src_image_data, torch.zeros_like(src_image_data), torch.zeros_like(src_image_data)))

        return src_image_data_threechannel, torch.unsqueeze(label_image_data, dim=0)



image_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            ToTensorV2(),
        ],
    )


def test_imagedataset():
    dataset = ImageDataset(train_src_image_folder, train_label_image_folder, image_transform)
    src_image, label_image = dataset[10]
    print(src_image)
    print(src_image.shape)
    print(label_image)
    print(label_image.shape)
    src_image_data = src_image[0]
    plt.imshow(src_image_data, cmap = "gray")
    plt.show()
    plt.imshow(label_image.squeeze(0), cmap="gray")
    plt.show()

def test_train_one():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder, image_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = models.segmentation.fcn_resnet50(pretrained=False, progress=False, num_classes=1)

    train_iter = iter(train_loader)
    inputs, labels = train_iter.next()
    results = model.forward(inputs)
    outputs = results["out"]
    print(outputs.shape)
    print(labels.shape)


def train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder, image_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = models.segmentation.fcn_resnet50(pretrained=False, progress=False, num_classes=1)

    # loss
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for batch_index, (images, labels) in enumerate(train_loader):        
            # forward
            results = model(images)
            outputs = results["out"]
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

            # During test, we could break earlier
            # if batch_index > 40:
            #    break

    torch.save(model, os.path.join(current_location, "saved-resnet.pth"))


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder, image_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, "saved-resnet.pth"))
    model.eval()
    with torch.no_grad():
        target_index = 20
        index = 0
        for images, labels in test_loader:
            if index == target_index:
                results = model(images)
                outputs = results["out"]
                print(f"min={outputs.min()}")
                print(f"max={outputs.max()}")
                
                outputs_raw = outputs.squeeze(0).squeeze(0)

                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                print(outputs.shape)
                print(outputs)

                f, axarr = plt.subplots(2,2)
                image_data = images.squeeze(0)[0]
                axarr[0,0].imshow(image_data, cmap = "gray")
                axarr[0,1].imshow(outputs_raw, cmap = "gray")
                axarr[1,0].imshow(labels.squeeze(0).squeeze(0), cmap = "gray")
                axarr[1,1].imshow(outputs.squeeze(0).squeeze(0), cmap = "gray")
                plt.show()


            index += 1


if __name__ == "__main__":
    # test_imagedataset()
    # test_train_one()
    train_and_save_model()
    # test_model()

