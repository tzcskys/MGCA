import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import BertEncoder, ImageEncoder, GlobalEmbedding, LocalEmbedding
from torch import distributed as dist

from bert_m3ae import BertCrossLayer, BertConfig, MIMHead
import warnings
import hashlib
import urllib
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import re


torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# os.environ['CUDA_VISIBLE_DEVICES']='1'

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
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)[0]
        return self.attn(x, x, x, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        # x = x + self.attention(self.ln_1(x), x_mask)
        out = self.attention(self.ln_1(x), x_mask)
        x = x + out[0]
        x = x + self.mlp(self.ln_2(x))
        return x, out[1]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers - 1)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None, return_attention=False):
        for i, block in enumerate(self.resblocks):
            if i < len(self.resblocks) - 1:
                x, _ = block(x, x_mask)
            else:
                # return attention map of the last block
                x, att = block(x, x_mask)
        if return_attention:
            return x, att
        else:
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

    def forward(self, x: torch.Tensor, x_mask=None, return_attention=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        t = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([t, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if return_attention:
            x, attention_map = self.transformer(x, x_mask, return_attention=True)
        else:
            x = self.transformer(x, x_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        if return_attention:
            return x, attention_map
        else:
            return x

    def forward_patch_embed(self, x: torch.Tensor, x_mask=None): # width here is like the feature dim, this function is to get the patch embedding
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




class ImageEncoder_m3ae(nn.Module):
    def __init__(self,
                 name: str = "ViT-B/16",
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 resolution_after: int = 224,
                 jit: bool = False
                 ):
        super(ImageEncoder_m3ae, self).__init__()

        self.model_name = name
        self.output_dim = output_dim # dim after teh projection

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

        self.feature_dim = vision_width

        self.model = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,  # no use, this is for text
            resolution_after=resolution_after,
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        model_dict = self.model.state_dict()
        pretrained_dict = state_dict
        if resolution_after != image_resolution:
            pretrained_dict = adapt_position_encoding(pretrained_dict, after=resolution_after,
                                                      patch_size=vision_patch_size)
        # 1. filter out unnecessary keys
        # this is because the clip downloaded model has text transformer outside, and all vit params in under "visual."
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith("visual.")}
        pretrained_dict = {k.replace("visual.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

        # projection layer
        self.global_embed = GlobalEmbedding(
            vision_width, hidden_dim, output_dim
        )

        self.local_embed = LocalEmbedding(
            vision_width, hidden_dim, output_dim
        )

    def forward(self, x, get_local=False, return_attention=False):
        if return_attention:
            img_feat, att = self.model(x,return_attention=True)
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous(), att
        else:
            img_feat = self.model(x)
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class m3aeAttMgca(LightningModule):
    '''Pytorch lightning implementation of m3aeAttMgca'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        # self.img_encoder_q = ImageEncoder(model_name=img_encoder, output_dim=self.hparams.emb_dim) # the raw mgca vit implementation does not contain attention_mask operation
        self.img_encoder_q = ImageEncoder_m3ae("ViT-B/16")
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)

        self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)
        if self._use_ddp_or_dpp2(self.trainer):
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        ## add for mim
        self.attention_mask_threshold = kwargs['attention_mask_threshold']
        self.mim_prob = kwargs['mim_prob']
        self.mim_layer = kwargs['mim_layer'] # layer index used to reconstrcut the image, start from 0
        self.batch_mask_ratio = kwargs['batch_mask_ratio']

        ## multi-modal fusion module for m3ae before mlm and mim
        bert_config = BertConfig(
            vocab_size=kwargs["vocab_size"],
            hidden_size=kwargs["hidden_size"],
            num_hidden_layers=kwargs["num_hidden_layers"],
            num_attention_heads=kwargs["num_attention_heads"],
            intermediate_size=kwargs["hidden_size"] * kwargs["mlp_ratio"],
            max_position_embeddings=kwargs["max_position_embeddings"],
            hidden_dropout_prob=kwargs["hidden_dropout_prob"],
            attention_probs_dropout_prob=kwargs["attention_probs_dropout_prob"],
        )

        # BertCrossLayer is a class that represents the cross-attention layer, takes as input the queries and keys from the current sequence and the values from the other sequence
        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(kwargs["num_hidden_layers"])])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(kwargs["num_hidden_layers"])])
        self.multi_modal_language_layers.apply(init_weights)

        mim_config = {
            "hidden_size": kwargs["hidden_size"],
            "patch_size": kwargs["patch_size"],
            "image_size": kwargs["image_size"],
            "mim_decoder_hidden_size": kwargs["mim_decoder_hidden_size"],
            "mim_decoder_num_layers": kwargs["mim_decoder_num_layers"],
            "mim_decoder_num_heads": kwargs["mim_decoder_num_heads"],
        }

        self.modality_type_embeddings = nn.Embedding(2, 128)
        self.modality_type_embeddings.apply(init_weights)

        self.mim_head = MIMHead(mim_config)
        self.mim_head.apply(init_weights)

        self.norm_pix_loss = kwargs['norm_pix_loss']


    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Forward of query image encoder
        # img_feat_q, patch_feat_q = self.img_encoder_q(batch["imgs"])
        img_feat_q, patch_feat_q, img_last_attn = self.img_encoder_q(batch["imgs"], return_attention=True) # img_last_attn: bz*12*197*197, 197=14*14(patch_num)+1; 12 is head num
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_ita = loss0 + loss1

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        ########### Token-level alignment ################
        # cross attention patch to sentences
        mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(
            batch["imgs"]).bool()

        if self.hparams.use_local_atten:
            word_atten_output, _ = self.word_local_atten_layer(
                word_emb_q, patch_emb_q, patch_emb_q)
        else:
            atten_sim = torch.bmm(word_emb_q, patch_emb_q.permute(0, 2, 1))
            word_num = word_emb_q.size(1)
            # atten_sim[mask.unsqueeze(1).repeat(1, word_num, 1)] = float("-inf")
            atten_scores = F.softmax(
                atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
            word_atten_output = torch.bmm(atten_scores, patch_emb_q)

        word_atten_output = F.normalize(word_atten_output, dim=-1)

        word_sim = torch.bmm(
            word_emb_q, word_atten_output.permute(0, 2, 1)) / self.hparams.local_temperature

        with torch.no_grad():
            atten_weights = word_attn_q.detach() # bz, 111
            word_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight.clone())
            word_atten_weights = torch.stack(word_atten_weights)
            # TODO: maybe clip the tensor of 10 percentile and 90 percentile

        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)

        word_sim = torch.bmm(word_emb_q, word_atten_output.permute(
            0, 2, 1)) / self.hparams.local_temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(
            word_emb_q).long().repeat(bz)
        loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
            word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        loss_word = (loss_word_1 + loss_word_2) / 2.

        if self.hparams.bidirectional:
            # Try not use atten layer
            if self.hparams.use_local_atten:
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb_q, word_emb_q.permute(0, 2, 1))
                patch_num = patch_emb_q.size(1)
                atten_sim[mask.unsqueeze(1).repeat(
                    1, patch_num, 1)] = float("-inf")
                atten_scores = F.softmax(
                    atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
                patch_atten_output = torch.bmm(atten_scores, word_emb_q)

            # patch_atten_output: bz, 196, 128
            if "vit" not in self.hparams.img_encoder:
                patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                patch_num = patch_atten_output.size(1)
                patch_atten_weights = torch.ones(
                    bz, patch_num).type_as(batch["imgs"]) / patch_num

            else:
                with torch.no_grad():
                    # img_attn_map = self.img_encoder_q.model.blocks[-1].attn.attention_map.detach() # bz, head_num(12), 197, 197; 197=patch_num(14*14)+[CLS]
                    img_attn_map = img_last_attn.detach().clone()
                    atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1) # bz*196
                    patch_atten_weights = []
                    for i in range(bz):
                        atten_weight = atten_weights[i]
                        atten_weight = atten_weight.clip(torch.quantile(
                            atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                        patch_atten_weights.append(atten_weight.clone())
                    patch_atten_weights = torch.stack(patch_atten_weights)

                patch_atten_weights /= patch_atten_weights.sum(
                    dim=1, keepdims=True)

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / self.hparams.local_temperature
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(
                patch_emb_q).long().repeat(bz)
            # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_local = loss_patch + loss_word

        else:

            loss_local = loss_word

        # normalize prototype layer
        with torch.no_grad():
            w = self.prototype_layer.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototype_layer.weight.copy_(w)

        # Compute assign code of images
        img_proto_out = self.prototype_layer(img_emb_q)
        report_proto_out = self.prototype_layer(report_emb_q)

        # TODO: define this to hparams
        with torch.no_grad():
            img_code = torch.exp(
                img_proto_out / self.hparams.epsilon).t()
            img_code = self.get_assignments(
                img_code, self.hparams.sinkhorn_iterations)         # bz, 500
            report_code = torch.exp(
                report_proto_out / self.hparams.epsilon).t()
            report_code = self.get_assignments(
                report_code, self.hparams.sinkhorn_iterations)       # bz, 500

        img_proto_prob = F.softmax(
            img_proto_out / self.hparams.proto_temperature, dim=1)
        report_proto_prob = F.softmax(
            report_proto_out / self.hparams.proto_temperature, dim=1)

        loss_i2t_proto = - \
            torch.mean(
                torch.sum(img_code * torch.log(report_proto_prob), dim=1))
        loss_t2i_proto = - \
            torch.mean(torch.sum(report_code *
                       torch.log(img_proto_prob), dim=1))

        loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.


        ## add the mim
        ret = {}

        features_attn_a = patch_emb_q  # [72,196,128], bz * patch_num * dim

        # get the cross-attention mask
        features_attn_b = report_emb_q # bz * dim

        # # get the self-attention mask
        # features_attn_b = img_emb_q

        features_attn_b = features_attn_b.unsqueeze(2)  # bz * dim * 1
        attn = torch.bmm(features_attn_a, features_attn_b)  # bz * patch_num * 1
        attn = attn.squeeze(2)  # bz * patch_num, [72, 196]
        attn = nn.Softmax(dim=-1)(attn * 1.0)

        N_max = int(attn.shape[1] * self.attention_mask_threshold)
        cross_attention_itm = torch.zeros(attn.shape, dtype=torch.bool, device=attn.device)

        # ## for MST, lowest N_max patches with the lowest attention scores will be randomly masked
        # idx = torch.argsort(attn, descending=False)[:,:N_max] # ascending order, dim default is -1, pick the lower N_max patches and set them to be True, and they will be randomly masked
        # cross_attention_itm.scatter_(1, idx, True)

        ## for AttMsk, highest N_max patches with the highest attention scores will be randomly masked
        idx = torch.argsort(attn, descending=True)[:,:N_max]  # descending order, dim default is -1, pick the top N_max patches and set them to be True, and they will be randomly masked
        cross_attention_itm.scatter_(1, idx, True)

        uni_modal_image_feats = self.img_encoder_q.model.forward_patch_embed(batch["imgs"]) # bz, patch_num+1, dim_before_proj; [72, 197, 768]
        ## if following AttMask, keep some images in each batch unmasked, and mask out the rest, attention_mask_image should be [bz, 197]
        uni_modal_image_feats, mim_masks, attention_mask_image, mim_ids_restore = self.random_masking_attnmask_batchmask(uni_modal_image_feats, self.mim_prob, self.attention_mask_threshold, cross_attention_itm, self.batch_mask_ratio)

        uni_modal_image_feats_out = self.img_encoder_q.model.forward_trans(uni_modal_image_feats, attention_mask_image)  # forward patches through transformer layers
        uni_modal_image_feats_global = uni_modal_image_feats_out[:,0].contiguous()
        uni_modal_image_feats_local = uni_modal_image_feats_out[:,1:].contiguous()
        attention_mask_image_len = attention_mask_image.shape[1] - attention_mask_image.sum(dim=1)  # record the length of the image after masking, CLS token is included

        uni_modal_image_feats_global = self.img_encoder_q.global_embed(uni_modal_image_feats_global)
        uni_modal_image_feats_global = F.normalize(uni_modal_image_feats_global, dim=-1)
        uni_modal_image_feats_local = self.img_encoder_q.local_embed(uni_modal_image_feats_local)
        uni_modal_image_feats_local = F.normalize(uni_modal_image_feats_local, dim=-1)

        uni_modal_image_feats = torch.cat([uni_modal_image_feats_global.unsqueeze(1), uni_modal_image_feats_local], dim=1) # bz * 197 * 128

        image_masks = torch.ones((uni_modal_image_feats_out.size(0), uni_modal_image_feats_out.size(1)),dtype=torch.long).cuda()  # cross-attention mask for image features, set 1 for image part
        ## !!! if we use attnmask_batchmask in masked image modelling, the uni_modal_image_feats_ contains all tokens including the CLS and masked out token, so the later part of uni_modal_image_feats_ are masked out token and we should set their mask to 0 before used for fusion module
        for sss in range(uni_modal_image_feats_out.size(0)):
            image_masks[sss, int(attention_mask_image_len[sss].item()):] = 0  # this also include the [CLS] token
        device = batch["imgs"].device
        extended_image_masks = self.get_extended_attention_mask(image_masks, image_masks.size(), device)

        uni_modal_text_feats = word_emb_q.clone()  # batch * max_len * embed_dim ## TODO: do I need to concat the CLS token???

        text_masks = torch.ones((uni_modal_text_feats.size(0),uni_modal_text_feats.size(1)),dtype=torch.long).cuda() # do not include CLS token
        for ttt in range(uni_modal_text_feats.size(0)):
            sent = sents[ttt] # include the CLS token
            pad_token_num = sent.count('[PAD]')
            assert len(sent) == uni_modal_text_feats.size(1) + 1
            text_masks[ttt, -pad_token_num:] = 0  # set the mask of the [PAD] token to 0

        extended_text_masks = self.get_extended_attention_mask(text_masks, text_masks.size(), device)

        ## assign type embeddings
        image_token_type_idx = 1  # just to distinguish the image token from the text token
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
            # using the torch.full_like() function to create a tensor of the same shape and dtype as the image_masks tensor, which presumably contains binary masks for the input image.
            # The values of the new tensor are set to image_token_type_idx, which is presumably an integer value indicating the modality type of the input image.
        )

        ## cross-attention for multimodal fusion
        x, y = uni_modal_text_feats, uni_modal_image_feats
        for layer_idx, (text_layer, image_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                  self.multi_modal_vision_layers)):
            if self.mim_layer == layer_idx:
                ret[f"multi_modal_text_feats_{layer_idx}"], ret[
                    f"multi_modal_image_feats_{layer_idx}"] = x, y  # we choose extract the 3rd layer feature for mim, following the m3ae paper, set in the config file

            x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
            y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]

        multi_modal_image_feats = ret[f"multi_modal_image_feats_{self.mim_layer}"]

        ret["mim_ids_restore"] = mim_ids_restore
        ret["mim_mask"] = mim_masks
        ret["patched_images"] = self.patchify(batch["imgs"], 16)

        mim_logits = self.mim_head(multi_modal_image_feats, ret["mim_ids_restore"], attention_mask_image_len)

        target = ret["patched_images"]
        if self.norm_pix_loss:
            mean = torch.mean(target, dim=-1, keepdim=True)
            var = torch.var(target, dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        mim_labels = target
        mask = ret["mim_mask"]

        loss_mim = (mim_logits - mim_labels) ** 2
        loss_mim = loss_mim.mean(dim=-1)  # [N, L], mean loss per batch
        loss_mim = (loss_mim * mask).sum() / mask.sum()  # mean loss on removed patches


        # return loss_ita, loss_local, loss_proto, acc1, acc5
        return loss_ita, loss_local, loss_proto, loss_mim, acc1, acc5

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            ## here I deleted the decoder part for simplicity, check https://huggingface.co/transformers/v4.11.3/_modules/transformers/modeling_utils.html#ModuleUtilsMixin.get_extended_attention_mask
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility # comment out since no self.dtype
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def patchify(self, imgs, p):
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def random_masking_attnmask_batchmask(self, x, mask_ratio, attn_mask_ratio, cross_attn_mask, batch_mask_ratio):
        """
        compared to random_masking(), this function will randomly mask from the selected patches, e.g., whose cross-attention in itm is lower than a threshold
        cross_attn_mask: [batch_size, num_patches], True for selected patches which will be randomly masked, False for unselected patches which will not be masked
        batch_mask_ratio: some samples in a batch will be masked, others will not be masked, following the AttMask to let the model see both masked and unmasked samples
        """
        assert x.shape[0] == cross_attn_mask.shape[0]
        assert x.shape[1] == cross_attn_mask.shape[1] + 1 # +1 for cls token

        x_ = x[:, :1] # cls token
        x = x[:, 1:]
        pos_embed = self.img_encoder_q.model.positional_embedding.unsqueeze(0).to(x) ## TODO: this is learned pos_emb for itm, should I reuse??

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio * attn_mask_ratio))

        # batch_mask = torch.randn(N, device=x.device)
        batch_mask = torch.rand(N, device=x.device)
        len_keep_batchmask = [L if batch_mask[i]>batch_mask_ratio else len_keep for i in range(N)]

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # filter out the unselected patches which will not be masked
        noise[cross_attn_mask==False] = -1.0

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)


        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = ids_shuffle[:, :L] # keep all patches, we will use attention_mask to let the model only attend to the unmasked patches

        x += pos_embed[:, 1:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove; this mask is used for the mlm loss to only compute the loss on the masked tokens
        mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        for i in range(N):
            mask[i, :len_keep_batchmask[i]] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        ## generate the attention_mask, this is for the transformer to only attend to the unmasked tokens, in bert_m3ae.py line 1860, 1 will be ignored in the attention computation
        attn_mask = torch.ones([N, L+1], device=x.device)
        for i in range(N):
            attn_mask[i, :len_keep_batchmask[i]+1] = 0 # +1 for cls token

        # append cls token
        x_ = x_ + pos_embed[:, :1]
        x_masked = torch.cat((x_, x_masked), dim=1)

        return x_masked, mask, attn_mask, ids_restore

    def sinkhorn(self, Q, nmb_iters):
        '''
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.hparams.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def training_step(self, batch, batch_idx):
        loss_ita, loss_local, loss_proto, loss_mim, acc1, acc5 = self(batch, batch_idx, "train")

        loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_proto + 1 * loss_mim

        log = {
            "train_loss": loss,
            "train_loss_ita": self.hparams.lambda_1 * loss_ita,
            "train_loss_local": self.hparams.lambda_2 * loss_local,
            "train_loss_proto": self.hparams.lambda_3 * loss_proto,
            "train_loss_mim": 1 * loss_mim,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss


    # freeze prototype layer
    def on_after_backward(self):
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            for param in self.prototype_layer.parameters():
                param.grad = None

    def validation_step(self, batch, batch_idx):
        loss_ita, loss_local, loss_proto, loss_mim, acc1, acc5 = self(
            batch, batch_idx, "valid")

        loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_proto + 1 * loss_mim

        log = {
            "val_loss": loss,
            "val_loss_ita": self.hparams.lambda_1 * loss_ita,
            "val_loss_local": self.hparams.lambda_2 * loss_local,
            "val_loss_proto": self.hparams.lambda_3 * loss_proto,
            "val_loss_mim": 1 * loss_mim,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss



    # def on_train_epoch_end(self):
    #     ''' Save img_queue and report_queue for visualization '''
    #     if self.local_rank == 0:
    #         img_queue_path = f"{self.trainer.callbacks[-1].dirpath}/img_queue.pth"
    #         torch.save(self.img_queue, img_queue_path)
    #         report_queue_path = f"{self.trainer.callbacks[-1].dirpath}/report_queue.pth"
    #         torch.save(self.report_queue, report_queue_path)

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,default=128, help="128, 256") # this is the proj dim
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)
        parser.add_argument("--num_prototypes", type=int, default=500)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1.)
        parser.add_argument("--lambda_2", type=float, default=1.)
        parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        ## add for mim
        parser.add_argument("--attention_mask_threshold",type=float,default=0.5)
        parser.add_argument("--batch_mask_ratio",type=float,default=2.0)
        parser.add_argument("--vocab_size",type=int,default=28996) # this is same as mgca text_encoder_q: BioClinicalBert
        parser.add_argument("--hidden_size",type=int,default=128) # TODO: m3ae use 768 since the feature dim is 768
        parser.add_argument("--num_hidden_layers",type=int,default=4)
        parser.add_argument("--num_attention_heads",type=int,default=4) #  hidden_size % num_attention_heads == 0; m3ae use 768 (dim) % 12 (head num)
        parser.add_argument("--mlp_ratio",type=int,default=4)
        parser.add_argument("--max_position_embeddings",type=int,default=512)
        parser.add_argument("--hidden_dropout_prob",type=float,default=0.1)
        parser.add_argument("--attention_probs_dropout_prob",type=float,default=0.1)
        parser.add_argument("--patch_size",type=int,default=16)
        parser.add_argument("--image_size",type=int,default=224)
        parser.add_argument("--mim_decoder_hidden_size",type=int,default=384) # TODO: just keep same as m3ae, MIMHead has a fc layer to first project to 384
        parser.add_argument("--mim_decoder_num_layers",type=int,default=4)
        parser.add_argument("--mim_decoder_num_heads",type=int,default=6)
        parser.add_argument("--mim_layer",type=int,default=3)
        parser.add_argument("--mim_prob",type=float,default=0.75)
        parser.add_argument("--norm_pix_loss",type=bool,default=True)

        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = m3aeAttMgca.add_model_specific_args(parser)
    args = parser.parse_args()

    # args.deterministic = True
    args.max_epochs = 50

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = m3aeAttMgca(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/m3aeAttMgca/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="m3aeAttMgca", save_dir=logger_dir, name=extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
