import torch
import torch.nn as nn
import torch.nn.functional as F
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
                                    transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

        src_image_data = preprocess(src_image)


        trf_resize = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)
        label_img_loc = os.path.join(self.label_image_folder, self.src_img_names[index])
        label_image = Image.open(label_img_loc).convert("L")
        label_image_resized = trf_resize(label_image)
        # create numpy data from the image
        label_image_data = np.array(label_image_resized)
        # we will then create two classes, the first is the background and the second is the label.
        defect_class = np.zeros_like(label_image_data)
        defect_class[label_image_data > 0] = 1.0
        # background class
        background_class = np.zeros_like(label_image_data)
        background_class[label_image_data == 0] = 1.0
        label_image_classes = torch.tensor([background_class, defect_class])

        # return
        # src_image_data: [3, W, H]. 3 means 3 channels
        # label_image_classes: [2, W, H]. The first element is background_class, the second element is defect_class
        return src_image_data, label_image_classes


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        # the model just use the deeplabv3
        self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=2)

    """
    x is [batch_size, 3, W, H]
    return is [batch_size, 2, W, H], where [batch_size, 0] is background_class, [batch_size, 1] is defect_class
    """
    def forward(self, x):
        results = self.dl(x)
        # as the deeplabv3 returns a dict, we will get the "out" from the dictionary
        ret = results["out"]
        return ret

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, targets):
        N = targets.size(0)
        inputs = torch.sigmoid(inputs)
        # flattern labels
        inputs = inputs.view(N, -1)
        targets = targets.view(N, -1)

        EPSILON = 1e-7

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2.0 * intersection + EPSILON)/(union + EPSILON)
        return 1 - dice/N


def calculate_loss(predicts, labels, criterion):
    # outputs size is [batch_size, 2, W, H]
    # We only need to use the defect class to calculate the IoU
    # Get the predicted defect class
    defect_preds = predicts[:, 1, :, :]
    # Get the targeted defect class
    defect_targets = labels.float()[:, 1, :, :]
    loss = criterion(defect_preds, defect_targets)
    return loss


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

    model = SegmentationModel()

    train_iter = iter(train_loader)
    inputs, labels = train_iter.next()
    predicts = model.forward(inputs)
    print(predicts.shape)
    print(labels.shape)

    criterion = DiceLoss()

    loss = calculate_loss(predicts, labels, criterion)
    print(loss)


def train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegmentationModel()

    # loss
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = DiceLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train_loop(train_loader, model, criterion, optimizer)

def load_checkpoint_and_train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegmentationModel()

    # loss
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    checkpoint = torch.load(os.path.join(current_location, "saved-deeplab-dice-checkpoint.pth"))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loop(train_loader, model, criterion, optimizer)


def train_loop(train_loader, model, criterion, optimizer):
    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for batch_index, (images, labels) in enumerate(train_loader):        
            # forward
            predicts = model(images)
            loss = calculate_loss(predicts, labels, criterion)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 10 == 0:
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

            # During test, we could break earlier
            # if batch_index > 40:
            #   break

    torch.save(model, os.path.join(current_location, "saved-deeplab-dice.pth"))
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, os.path.join(current_location, "saved-deeplab-dice-checkpoint.pth"))


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, "saved-deeplab-dice.pth"))
    model.eval()
    criterion = DiceLoss()
    with torch.no_grad():
        # only show the one that we are interesting
        # 20, 30 are OK, but
        # 40, 42 are not OK.
        target_index = 20
        index = 0
        for image, label in test_loader:
            if index == target_index:
                output = model(image)
                print(f"max={output.max()}")
                print(f"min={output.min()}")

                loss = calculate_loss(output, label, criterion)
                print(f"loss={loss}")

                output_raw = output.squeeze(0)
                print(f"max={output_raw.max()}")
                print(f"min={output_raw.min()}")

                output_background = output_raw[0]
                output_defect = output_raw[1]

                # the index who has the max value. As we only have two classes, it's either 0 or 1
                output_label = output_raw.argmax(dim=0)
                print(output_label.shape)
                print(output_label)


                # draw the images
                f, axarr = plt.subplots(2,3)
                image_data = image.squeeze(0)[0]
                target_label = label.squeeze(0)[1]

                intersection = output_label.logical_and(target_label)
                union = output_label.logical_or(target_label)
                accuracy = 1.0
                if union.sum().item() != 0:
                    accuracy = intersection.sum().item() / union.sum().item()
                print(f"index={index}: accuracy={accuracy}")


                # show the image
                axarr[0,0].imshow(image_data, cmap = "gray")

                # background
                axarr[0,1].imshow(torch.sigmoid(output_raw[0]), cmap="gray")

                # defect
                axarr[0,2].imshow(torch.sigmoid(output_raw[1]), cmap="gray")

                # expected label
                axarr[1,0].imshow(target_label, cmap = "gray")

                # defect label
                axarr[1,1].imshow(output_label, cmap = "gray")
                # show ytest, it should be same as output_label
                # axarr[1,2].imshow(ytest, cmap="gray")
                plt.show()


            index += 1

def test_model_accuracy():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, "saved-deeplab-dice.pth"))
    model.eval()
    criterion = DiceLoss()
    with torch.no_grad():
        index = 0
        accuracy_list = []

        for image, label in test_loader:
            output = model(image)

            loss = calculate_loss(output, label, criterion)
            print(f"loss={loss}")

            output_raw = output.squeeze(0)

            # the index who has the max value. As we only have two classes, it's either 0 or 1
            output_label = output_raw.argmax(dim=0)

            target_label = label.squeeze(0)[1]

            intersection = output_label.logical_and(target_label)
            union = output_label.logical_or(target_label)
            accuracy = 1.0
            if union.sum().item() != 0:
                accuracy = intersection.sum().item() / union.sum().item()
            accuracy_list.append(accuracy)
            print(f"index={index}: accuracy={accuracy}")

            index = index + 1

        accuracy_list_np = np.array(accuracy_list)
        print(f"Accuracy: min={accuracy_list_np.min()}, max={accuracy_list_np.max()}, average={np.average(accuracy_list_np)}")
        print(f"The least accuracy index={accuracy_list_np.argmin()}")



if __name__ == "__main__":
    # test_imagedataset()
    # test_train_one()
    # train_and_save_model()
    load_checkpoint_and_train_and_save_model()
    # test_model()
    # test_model_accuracy()

