import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth = 1):
        inputs = F.sigmoid(inputs)
        # flattern labels
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice



if __name__ == "__main__":
    a = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    print(a.shape)
    a = a.view(-1)
    print(a.shape)
    
