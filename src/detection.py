import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import os

train_src_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/srcImg"
train_label_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/label"
test_src_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/val_image_label/srcImg"
test_label_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/val_image_label/val_label"

batch_size = 2
image_size = 909 * 710
image_size_resized = 92 * 72
learning_rate = 0.001

class SourceImageAndLabelImageDataset(Dataset):
    def __init__(self, src_image_folder, label_image_folder, transform):
        super(SourceImageAndLabelImageDataset, self).__init__()
        self.src_image_folder = src_image_folder
        self.label_image_folder = label_image_folder
        self.transform = transform
        self.src_img_names = os.listdir(src_image_folder)

    def __len__(self):
        return len(self.src_img_names)

    def __getitem__(self, index):
        src_img_loc = os.path.join(self.src_image_folder, self.src_img_names[index])
        src_image = Image.open(src_img_loc)
        src_image = src_image.resize((72, 92), Image.ANTIALIAS)
        tensor_src_image = self.transform(src_image)

        label_img_loc = os.path.join(self.label_image_folder, self.src_img_names[index])
        label_image = Image.open(label_img_loc)
        label_image = label_image.resize((72, 92), Image.ANTIALIAS)
        tensor_lable_image = self.transform(label_image)
        return tensor_src_image, tensor_lable_image

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# model
model = NeuralNetwork(image_size_resized, image_size_resized)

# loss and optimizer
criterion = nn.MSELoss()
# use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



if __name__ == "__main__":
    train_dataset = SourceImageAndLabelImageDataset(train_src_image_folder, train_label_image_folder, transforms.ToTensor())
    print(len(train_dataset))
    src, label = train_dataset[3]
    print(src)
    print(src.size())
    print(label)

    test_dataset = SourceImageAndLabelImageDataset(test_src_image_folder, test_label_image_folder, transforms.ToTensor())
    print(len(test_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



    # training loop
    num_epochs = 1
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (src, label) in enumerate(train_loader):
            src = src.reshape(-1, image_size_resized)
            label = label.reshape(-1, image_size_resized)

            # forward
            outputs = model(src)
            loss = criterion(outputs, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"epoch {epoch}, step {i}/{n_total_steps}, loss={loss}")


    # Testing loop
    



