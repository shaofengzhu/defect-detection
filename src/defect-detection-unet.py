import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
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
num_epochs = 2

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

        return src_image_data, torch.unsqueeze(label_image_data, dim=0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature
        
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[:: -1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

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
    plt.imshow(src_image.squeeze(0), cmap = "gray")
    plt.show()
    plt.imshow(label_image.squeeze(0), cmap="gray")
    plt.show()

def test_UNET():
    x = torch.randn((2, 1, 920, 716))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)

    x = torch.randn((1, 920, 716))
    x = torch.unsqueeze(x, dim=0)
    preds = model(x)
    print(preds.shape)

def test_loader():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder, image_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    examples = iter(train_loader)
    samples, labels = examples.next()
    print(samples.shape, labels.shape)


def train_model():

    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder, image_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = UNET(in_channels=1, out_channels=1)


    # loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for batch_index, (images, labels) in enumerate(train_loader):        
            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            #backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

            # During test, we could break earlier
            # if batch_index > 40:
            #     break

    torch.save(model.state_dict(), os.path.join(current_location, "saved.pth"))


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder, image_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = UNET(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(os.path.join(current_location, "saved.pth")))
    model.eval()
    with torch.no_grad():
        target_index = 20
        index = 0
        for images, labels in test_loader:
            if index == target_index:
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                print(outputs.shape)
                print(outputs)

                f, axarr = plt.subplots(2,2)
                axarr[0,0].imshow(images.squeeze(0).squeeze(0), cmap = "gray")
                axarr[1,0].imshow(labels.squeeze(0).squeeze(0), cmap = "gray")
                axarr[1,1].imshow(outputs.squeeze(0).squeeze(0), cmap = "gray")
                plt.show()


            index += 1




if __name__ == "__main__":
    # test_UNET()
    # test_imagedataset()
    # test_loader()
    train_model()
    # test_model()
