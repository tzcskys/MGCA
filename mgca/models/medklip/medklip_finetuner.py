import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from mgca.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms, Moco2Transform
from mgca.models.mgca.mgca_module import MGCA
from mgca.models.ssl_finetuner import SSLFineTuner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# checked the code, the mgca version resnet50 is the same as the torchvision version
from torchvision.models import resnet50
import re
from torch import nn

class CNN(nn.Module):
    def __init__(self, backbone):
        super(CNN, self).__init__()

        self.cnn = resnet50()

    def forward(self, images):
        x = self.cnn.conv1(images)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = self.cnn.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x) # removed the last fc layer, and the SSLFineTuner will add a new fc layer

        return x, 0 # this is only because the SSLFineTuner requires 2 outputs


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--path", type=str, default="/mnt/Research/mingjian/results/pre_trained_model/mgca/MedKLIP_checkpoint_final.pth")
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
        model = CNN("resent50")
        state_dict = torch.load(args.path)
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

        model.cnn.load_state_dict(new_params, strict=False)  # strict has to set as false since we miss the last resblock, also not have the fc layer
    else:
        model = MGCA()

    args.model_name = "resnet_50"
    args.backbone = model
    args.in_features = 2048

    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    ## this is because the medklip pretrained model do not have the last resblock, so we cant fix that part.
    ## for medklip, the SSLFineTuner init will freeze the whole backbone, and the shared_step() of SSLFineTuner will also using "with torch.no_grad()"
    ## !! so we need to manually set the requires_grad=True for the last resblock (layer4) here after the SSLFineTuner init, and comment out the "with torch.no_grad()" from shared_step() of SSLFineTuner
    for name, param in tuner.backbone.cnn.named_parameters():
        if name.startswith("layer4."):
            param.requires_grad = True

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/medklip_finetune/{extension}")
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
        project="medklip_finetune",
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
