import torch
import torchvision.models as models

model = models.segmentation.fcn_resnet50(pretrained=False, progress=False, num_classes=1)

input = torch.randn((2,3,40,30))
output = model.forward(input)

for key in output:
    print(key)

out = output["out"]
print(out.shape)
print(out[0][0].min())
print(out[0][0].max())
print(out[0][0])