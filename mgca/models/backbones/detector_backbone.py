import ipdb
import torch
import torch.nn as nn
from mgca.models.backbones import cnn_backbones
from torch import nn


class ResNetDetector(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()

        model_function = getattr(cnn_backbones, model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=pretrained
        )

        if model_name == "resnet_50":
            self.filters = [512, 1024, 2048]

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        out3 = self.model.layer2(x)   # bz, 512, 28
        out4 = self.model.layer3(out3)
        out5 = self.model.layer4(out4)

        return out3, out4, out5


class VitDetector(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.proj = nn.Conv2d(768, 1024, 1)
        self.filters = [512, 1024, 1024] # the downsample use maxpooling so the filter is the same
        self.upsample2x = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.downsample2x = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.model(x) # bz, 197, 768
        x = x[:,1:,:] # remove cls token
        x = x.permute(0, 2, 1).contiguous() # bz, 768, 196
        patch_num = x.shape[-1]
        n_h = n_w = int(patch_num ** 0.5)
        x = x.reshape(x.shape[0], x.shape[1], n_h, n_w) # bz, 768, 14, 14
        x = self.proj(x) # bz, 1024, 14, 14

        ## following the ViTDet, use the last feature map to construct the simple FPN
        x_up = self.upsample2x(x) # bz, 512, 28, 28
        x_down = self.downsample2x(x) # bz, 512, 7, 7

        return x_up, x, x_down


if __name__ == "__main__":
    model = ResNetDetector("resnet_50")
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    ipdb.set_trace()
