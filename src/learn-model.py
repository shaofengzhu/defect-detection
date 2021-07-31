from matplotlib import image
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 10)

    def forward(self, x):
        z1 = self.l1(x)
        a1 = self.relu(z1)
        z2 = self.l2(a1)
        a2 = self.relu(z2)
        z3 = self.l3(a2)
        a3 = self.relu(z3)
        return a3

train_dataset = torchvision.datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)


def train_model():
    model = MyModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

    for epoch in range(30):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28)

            # forward
            y = model(images)

            # loss
            loss = loss_fn(y, labels)

            # backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                print(f"epoch {epoch} step {i}: loss={loss}")

    torch.save(model.state_dict(), "my-mnist.pth")

def test_model_one_by_one():
    model = MyModel()
    model.load_state_dict(torch.load("my-mnist.pth"))
    model.eval()

    inputStr = None
    inputStr = input("Type Image Index:")
    while len(inputStr) > 0:
        index = int(inputStr)
        image, label = test_dataset[index]
        image = image.reshape(-1, 28 * 28)
        output = model(image)
        print(f"label={label}")
        print(output)
        print(output.shape)
        pred = output.argmax(dim=1)
        print(f"pred={pred}")
        correct = pred.squeeze().item() == label
        print("Correct" if correct else "Incorrect")


        inputStr = input("Type Image Index:")


def test_model():
    model = MyModel()
    model.load_state_dict(torch.load("my-mnist.pth"))
    model.eval()

    count = 0
    correct_count = 0
    for index in range(len(test_dataset)):
        image, label = test_dataset[index]
        image = image.reshape(-1, 28 * 28)
        output = model(image)
        pred = output.argmax(dim=1)
        correct = pred.squeeze().item() == label
        count += 1
        if correct:
            correct_count += 1
    print(f"{correct_count} of {count} are correct. Accuracy={correct_count / count}")


def check_dataset():
    print(len(train_dataset))
    index = 10
    image, label = train_dataset[index]
    print(f'label={label}')
    print(image.shape)

    image_data = image.squeeze(dim=0)
    print(image_data.shape)

    plt.imshow(image_data, cmap="gray")
    plt.show()

if __name__ == "__main__":
    # train_model()
    test_model()


