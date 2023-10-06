import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from mgca.datasets.data_module import DataModule
from mgca.datasets.detection_dataset import (OBJCXRDetectionDataset,
                                             RSNADetectionDataset)
from mgca.datasets.transforms import DetectionDataTransforms
from mgca.models.backbones.detector_backbone import ResNetDetector, VitDetector
from mgca.models.ssl_detector import SSLDetector
from mgca.models.mrm.mrm_module import MRM

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cli_main():
    parser = ArgumentParser("Finetuning of object detection task for MGCA")
    parser.add_argument("--base_model", type=str, default="vit")
    parser.add_argument("--ckpt_path", type=str, default="/mnt/HDD2/mingjian/results/pre_trained_model/mgca/MRM.pth")
    # parser.add_argument("--ckpt_path", type=str, default="/mnt/HDD2/mingjian/results/pre_trained_model/mgca/resnet_50.ckpt")
    parser.add_argument("--dataset", type=str,
                        default="rsna", help="rsna or object_cxr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50
    args.accelerator = "gpu"

    seed_everything(args.seed)

    if args.dataset == "rsna":
        datamodule = DataModule(RSNADetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    elif args.dataset == "object_cxr":
        datamodule = DataModule(OBJCXRDetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    else:
        raise RuntimeError(f"{args.dataset} does not exist!")


    model_mrm = MRM(img_encoder="vit-base")  # in here means vit-B/16
    state_dict = torch.load(args.ckpt_path)
    params = {k: v for k, v in state_dict.items()}
    params = params["model"]
    params = {("img_encoder.model." + k): v for k, v in params.items()}
    params = {k: v for k, v in params.items() if k in model_mrm.state_dict()}
    model_mrm.load_state_dict(params, strict=False)

    args.img_encoder = VitDetector(model_mrm.img_encoder.model)

    # Freeze encoder
    for param in args.img_encoder.model.parameters():
        param.requires_grad = False

    model = SSLDetector(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/mrm_detection/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_mAP", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_mAP", min_delta=0.,
                      patience=10, verbose=False, mode="max")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="mrm_detection", save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger
    )
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
