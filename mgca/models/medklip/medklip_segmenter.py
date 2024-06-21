import datetime
import os
from argparse import ArgumentParser

import segmentation_models_pytorch as smp
import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from mgca.datasets.data_module import DataModule
from mgca.datasets.segmentation_dataset import (RSNASegmentDataset,
                                                SIIMImageDataset, TBX11KSegmentDataset)
from mgca.models.backbones.transformer_seg import SETRModel
# from mgca.models.mgca.mgca_module import MGCA
from mgca.models.ssl_segmenter import SSLSegmenter

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


def cli_main():
    parser = ArgumentParser(
        "Finetuning of semantic segmentation task for MGCA")
    parser.add_argument("--base_model", type=str,
                        default="resnet50", help="resnet50 or vit")
    parser.add_argument("--ckpt_path", type=str, default="/mnt/HDD2/mingjian/results/pre_trained_model/mgca/MedKLIP_checkpoint_final.pth")
    parser.add_argument("--dataset", type=str, default="siim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # args.deterministic = True
    args.max_epochs = 50
    args.accelerator = "gpu"

    seed_everything(args.seed)

    if args.dataset == "siim":
        datamodule = DataModule(SIIMImageDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNASegmentDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)
    elif args.dataset == "tbx11k":
        datamodule = DataModule(TBX11KSegmentDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)

    # mgca = MGCA.load_from_checkpoint(args.ckpt_path)
    # encoder = mgca.img_encoder_q.model

    if args.base_model == "vit":
        args.seg_model = SETRModel(
            patch_size=(16, 16),
            in_channels=3,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            decode_features=[512, 256, 128, 64]
        )
        args.seg_model.encoder_2d.bert_model = model

        for param in args.seg_model.encoder_2d.bert_model.parameters():
            param.requires_grad = False

    elif args.base_model == "resnet50":
        # FIXME: fix this later
        args.seg_model = smp.Unet(
            args.base_model, encoder_weights=None, activation=None)


        state_dict = torch.load(args.ckpt_path)
        params = {re.sub('^module.res_features.', '', k): v for k, v in state_dict["model"].items()}
        new_params = dict()
        # the medklip model has different naming convention, so we need to map it back; they do not have the last resblock
        # see https://github.com/MediaBrain-SJTU/MedKLIP/blob/main/PreTrain_MedKLIP/models/model_MedKLIP.py
        for k, v in params.items():
            if k.startswith("0"):
                new_k = "conv1" + k[1:]
                new_params[new_k] = v
            elif k.startswith("1"):
                new_k = "bn1" + k[1:]
                new_params[new_k] = v
            elif k.startswith("4"):
                new_k = "layer1" + k[1:]
                new_params[new_k] = v
            elif k.startswith("5"):
                new_k = "layer2" + k[1:]
                new_params[new_k] = v
            elif k.startswith("6"):
                new_k = "layer3" + k[1:]
                new_params[new_k] = v

        # the defined resnet50 has one fc layer; fc is not used in the segmentation, see the mgca_segmenter.py
        new_params["fc.bias"] = None
        new_params["fc.weight"] = None

        args.seg_model.encoder.load_state_dict(new_params, strict=False) # strict has to set as false since we miss the last resblock
        # Freeze encoder
        # we cannot freeze the last resblock, because it is not pretrained by medklip
        for param in args.seg_model.encoder.parameters():
            param.requires_grad = False

        for name, param in args.seg_model.encoder.named_parameters():
            if name.startswith("layer4."):
                param.requires_grad = True


    model = SSLSegmenter(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/medklip_segmentation/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="medklip_segmentation", save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    ## run test before train, to check if the model is loaded correctly
    trainer.test(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path='best')


if __name__ == "__main__":
    cli_main()
