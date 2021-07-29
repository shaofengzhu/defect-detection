import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

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
# make the size smaller so that we could run it quickly
image_width = 716 // 4
image_height = 920 // 4
learning_rate = 0.0001
num_epochs = 2

class ImageDataset(Dataset):
    def __init__(self, src_image_folder, label_image_folder):
        super(ImageDataset, self).__init__()
        self.src_image_folder = src_image_folder
        self.label_image_folder = label_image_folder
        self.src_img_names = os.listdir(src_image_folder)
        mod = len(self.src_img_names) % batch_size
        # remove the entries that not fall into the batch
        self.src_img_names = self.src_img_names[0: len(self.src_img_names) - mod]
        print(len(self.src_img_names))


    def __len__(self):
        return len(self.src_img_names)

    def __getitem__(self, index):
        src_img_loc = os.path.join(self.src_image_folder, self.src_img_names[index])
        src_image = Image.open(src_img_loc).convert("RGB")

        preprocess = transforms.Compose([
                                    transforms.Resize((image_height, image_width), 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

        src_image_data = preprocess(src_image)


        trf_resize = transforms.Resize((image_height, image_width), 2)
        trf_tensor = transforms.ToTensor()

        label_img_loc = os.path.join(self.label_image_folder, self.src_img_names[index])
        label_image = Image.open(label_img_loc).convert("L")

        # also resize the lable image and convert it to tensor
        y1 = trf_tensor(trf_resize(label_image))
        # convert it to boolean. If the value is greater than 0, it's the label, otherwise, it's the background
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        # now concat them together, the first one is background, the second is label
        y = torch.cat([y2, y1], dim=0)


        return src_image_data, y


class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        # the model just use the deeplabv3
        self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=2)

    def forward(self, x):
        results = self.dl(x)
        # as the deeplabv3 returns a dict, we will get the "out" from the dictionary
        ret = results["out"]
        return ret



def test_imagedataset():
    dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    src_image, label_image = dataset[10]
    print(src_image)
    print(src_image.shape)
    print(label_image)
    print(label_image.shape)

    fig, axarr= plt.subplots(2,3)

    # the image has three channels
    axarr[0,0].imshow(src_image[0], cmap = "gray")
    axarr[0,1].imshow(src_image[1], cmap = "gray")
    axarr[0,2].imshow(src_image[2], cmap = 'gray')

    # the labels has two, the first element is background, the second element is the label
    axarr[1,0].imshow(label_image[0], cmap="gray")
    axarr[1,1].imshow(label_image[1], cmap="gray")
    plt.show()

def test_train_one():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = SegModel()

    train_iter = iter(train_loader)
    inputs, labels = train_iter.next()
    outputs = model.forward(inputs)
    print(outputs.shape)
    print(labels.shape)

    targets = labels.float()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, targets)
    print(loss)


def train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegModel()

    # loss
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for batch_index, (images, labels) in enumerate(train_loader):        
            # forward
            predicts = model(images)

            # loss
            targets = labels.float()
            loss = criterion(predicts, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

            # During test, we could break earlier
            # if batch_index > 40:
            #   break

    torch.save(model, os.path.join(current_location, "saved-deeplab.pth"))


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, "saved-deeplab.pth"))
    model.eval()
    with torch.no_grad():
        # only show the one that we are interesting
        target_index = 20
        index = 0
        for image, label in test_loader:
            if index == target_index:
                output = model(image)
                
                output_raw = output.squeeze(0)

                output_background = output_raw[0]
                output_defect = output_raw[1]

                background_img = transforms.ToPILImage()(output_background)
                defect_image = transforms.ToPILImage()(output_defect)
                # we could show the image
                # background_img.show()
                # defect_image.show()

                # the index who has the max value. As we only have two classes, it's either 0 or 1
                output_label = output_raw.argmax(dim=0)
                print(output_label.shape)
                print(output_label)


                # if the defect class is larger than background, it's defect.
                # ytest should be same as output_label
                ytest = output_raw[1] > output_raw[0]

                # draw the images
                f, axarr = plt.subplots(2,3)
                image_data = image.squeeze(0)[0]
                label_data = label.squeeze(0)[1]

                # show the image
                axarr[0,0].imshow(image_data, cmap = "gray")

                # background
                axarr[0,1].imshow(output_raw[0], cmap="gray")

                # defect
                axarr[0,2].imshow(output_raw[1], cmap="gray")

                # expected label
                axarr[1,0].imshow(label_data, cmap = "gray")

                # defect label
                axarr[1,1].imshow(output_label, cmap = "gray")
                # show ytest, it should be same as output_label
                axarr[1,2].imshow(ytest, cmap="gray")
                plt.show()


            index += 1


if __name__ == "__main__":
    # test_imagedataset()
    # test_train_one()
    # train_and_save_model()
    test_model()

