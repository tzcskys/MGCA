import os
import torch
from argparse import ArgumentParser
from dateutil import tz
import datetime
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging
)
from mgca.datasets.data_module import DataModule
from mgca.datasets.classification_dataset import CheXpertImageDataset, RSNAImageDataset, COVIDXImageDataset
from mgca.datasets.transforms import DataTransforms, Moco2Transform
from mgca.models.ssl_finetuner import SSLFineTuner
from mgca.models.gloria.gloria_module import GLoRIA

import re
import hashlib
import urllib
import warnings
import tqdm
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torchvision.models import resnet50
from typing import Type, Union, Any, Callable, Union, List, Optional
import numpy as np
from torch import Tensor


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")
    return download_target

def adapt_position_encoding(model, patch_size=32, after=384, suffix='visual.positional_embedding'):
    keys = [k for k in model if k.endswith(suffix)]
    assert len(keys) == 1
    key = keys[0]
    origin_pos_embed = model[key]
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    grid_before = int(np.sqrt(origin_pos_embed.shape[1] - 1))
    before = int(grid_before * patch_size)
    assert (before % patch_size) == 0
    grid_after = after // patch_size
    assert (after % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = torch.nn.functional.interpolate(pos_embed.permute((2, 0, 1)).unsqueeze(0), size=new_size,
                                                mode='bicubic')
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    assert pos_embed.shape == (1, grid_after * grid_after + 1, embed_dim)
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    model[key] = pos_embed
    return model

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x_mask: torch.Tensor):
        if x_mask is not None:
            x_mask = x_mask.to(dtype=torch.bool, device=x.device)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)[0]

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), x_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers - 1)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for block in self.resblocks:
            x = block(x, x_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 resolution_after: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((resolution_after // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

        self.width = width

    def forward(self, x: torch.Tensor, x_mask=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        t = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([t, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, x_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        # return x
        return x[:, 0, :], x[:, 1:, :]  # this is just for fine-tuning in MGCA project!!

    def forward_patch_embed(self, x: torch.Tensor, x_mask): # width here is like the feature dim, this function is to get the patch embedding
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        t = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([t, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        return x

    def forward_pos_embed(self, x: torch.Tensor, x_mask):
        x = x + self.positional_embedding.to(x.dtype)
        return x

    def forward_trans(self, x: torch.Tensor, x_mask):
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, x_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x



def build_model(name, resolution_after=224, jit=False):
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}"
                           )
    try:
        model_clip = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
    state_dict = state_dict or model_clip.state_dict()

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    vision_heads = vision_width // 64

    model = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim, # no use, this is for text
            resolution_after=resolution_after,
    )

    # for key in ["input_resolution", "context_length", "vocab_size"]:
    #     if key in state_dict:
    #         del state_dict[key]
    #
    # model_dict = model.state_dict()
    # pretrained_dict = state_dict
    # if resolution_after != image_resolution:
    #     pretrained_dict = adapt_position_encoding(pretrained_dict, after=resolution_after, patch_size=vision_patch_size)
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)
    return model



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) ## be careful, the last fc layer used for classification will be removed in CNN(), and SSLFintuner will append a new fc layer
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
        raise NotImplementedError
    return model

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def get_network(backbone, output_layer, pretrained, weights=None, **kwargs):
    """
    Create sub-network given a backbone and an output_layer
    """
    # Create avgpool for densenet, doesnt exist as such
    if 'densenet' in backbone and output_layer == 'avgpool':
        sub_network = get_network(backbone, 'features', pretrained, **kwargs)
        sub_network.add_module('relu', nn.ReLU(inplace=True))
        sub_network.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        sub_network.add_module('flatten', nn.Flatten(1))
        return sub_network

    # torchxrayvision
    if "xrv" in backbone.lower():
        network = eval(backbone)(weights=weights)
        if hasattr(network, 'model'):  # XrvResNet Fix
            network = network.model
    else:
        network = eval(backbone)(pretrained=pretrained, **kwargs)

    if output_layer is not None and (not output_layer == 'classifier'): ## this removed the last raw fc layer from the resnet for downstream tasks, since we set output_layer to be "avgpool"
        layers = [n for n, _ in network.named_children()]
        assert output_layer in layers, '{} not in {}'.format(output_layer, layers)
        sub_network = []
        for n, c in network.named_children():
            sub_network.append(c)
            if n == output_layer: # break the loop after adding the user specified output layer, e.g., avgpool, and the last fc layer will be removed
                break
        network = nn.Sequential(*sub_network)

    return network

class CNN(nn.Module):
    def __init__(self, backbone, dropout_out, permute, freeze=True, output_layer=None, pretrained=True,
                 visual_embedding_dim=None, **kwargs):
        super(CNN, self).__init__()
        self.backbone = backbone
        self.output_layer = output_layer
        self.permute = permute
        self.freeze = freeze
        self.pretrained = pretrained

        self.cnn = get_network(self.backbone, self.output_layer, self.pretrained, **kwargs)
        self.dropout_out = nn.Dropout(p=dropout_out)

        assert permute in ["batch_first", "spatial_first", "no_permute"]

        if freeze:
            for name, param in self.cnn.named_parameters():
                param.requires_grad = False

    def forward(self, images, **kwargs):
        out = self.cnn(images)
        out = self.dropout_out(out)
        # if self.permute == "no_permute":
        #     out = out
        # elif self.permute == "batch_first":
        #     out = out.view(*out.size()[:2], -1).permute(0, 2, 1) # (batch, H*W, C)
        #     if out.shape[1] == 1:  # avgpool case
        #         out = out.squeeze(1)
        # elif self.permute == "spatial_first":
        #     out = out.view(*out.size()[:2], -1).permute(2, 0, 1)
        # else:
        #     raise NotImplementedError()

        return out, 0

    def train(self, mode: bool = True):
        if self.freeze:
            mode = False
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def __repr__(self):
        s = str(self.backbone) + '(output_layer=' + self.output_layer + ', dropout_out=' + str(
            self.dropout_out.p) + ', freeze=' + str(self.freeze) + ', pretrained=' + str(self.pretrained) + \
            ('\n classifier= {}'.format(self.cnn.classifier) if self.output_layer == 'classifier' else '' + ')')
        return s

def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    # parser.add_argument("--path", type=str, default="/mnt/HDD2/mingjian/results/pre_trained_model/mgca/14.48168_11_439185.pth")
    parser.add_argument("--path", type=str, default="/mnt/HDD2/mingjian/results/pre_trained_model/mgca/4.922432_10_621951.pth")
    parser.add_argument("--base_model", type=str, default="vit", help="resnet50 or vit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_pct", type=float, default=0.01)
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.dataset == "chexpert":
        # define datamodule
        # check transform here
        datamodule = DataModule(CheXpertImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 5
        multilabel = True
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNAImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 1
        multilabel = True
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")


    if args.path:
        if args.base_model == "vit":
            model = build_model("ViT-B/16")

            state_dict = torch.load(args.path)
            # the vision_encoder.ln_final, vision_encoder.token_embedding, vision_encoder.positional_embedding are not used since it is used by m3ae for text, no relation to the img
            # but the vision_encoder.visual.positional_embedding is used
            params = {re.sub('^vision_encoder.visual.', '', k): v for k, v in state_dict["model"].items()} # this is for m3ae gloria pre-trained model
            # params = {re.sub('^module.vision_encoder.visual.', '', k): v for k, v in state_dict["model"].items()} # this is for m3ae MST pre-trained model
            params = {k: v for k, v in params.items() if k in model.state_dict()}
            model.load_state_dict(params, strict=True)
        elif args.base_model == "resnet50":
            model = CNN("resnet50", 0.0, "batch_first", False, "avgpool", False)
            state_dict = torch.load(args.path)
            params = {re.sub('^visual.cnn.', '', k): v for k, v in state_dict["model"].items()}
            params = {k: v for k, v in params.items() if k in model.cnn.state_dict()}
            model.cnn.load_state_dict(params, strict=True)
        else:
            raise RuntimeError(f"no base model called {args.base_model}")
    else:
        # model = GLoRIA()
        raise RuntimeError("no path provided")

    # args.model_name = model.hparams.img_encoder
    # args.backbone = model.img_encoder
    # args.in_features = args.backbone.feature_dim

    args.model_name = "vit_m3ae_version"
    args.backbone = model # TODO
    if args.base_model == "vit":
        args.in_features = args.backbone.width # TODO
    else:
        args.in_features = 2048 # output dim of resnet 50 last layer

    args.num_classes = num_classes
    args.multilabel = multilabel


    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/gloria_finetune/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data/wandb")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="gloria_finetune",
        save_dir=logger_dir, 
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()