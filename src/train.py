# Training script for coastline segmentation
# Conor O'Sullivan

import numpy as np
import random
import glob
import argparse
import os
import json
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network import get_model
from dataset import TrainDataset
from utils import training_data_check


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_data(args):
    train_paths = glob.glob(os.path.join(args.train_path, "*.npy"))
    print(f"Total training images: {len(train_paths)}")
    training_data_check(train_paths, args)

    if args.sample:
        train_paths = train_paths[:100]

    random.seed(args.seed)
    random.shuffle(train_paths)

    if args.valid_path is not None:
        print("Using explicit validation set:", args.valid_path)
        valid_paths = glob.glob(os.path.join(args.valid_path, "*.npy"))
        train_dataset = TrainDataset(train_paths, args)
        valid_dataset = TrainDataset(valid_paths, args)
    else:
        split_idx = int(args.split * len(train_paths))
        train_dataset = TrainDataset(train_paths[:split_idx], args)
        valid_dataset = TrainDataset(train_paths[split_idx:], args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              worker_init_fn=lambda wid: np.random.seed(args.seed + wid))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    return train_loader, valid_loader


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

def train_model(train_loader, valid_loader, args):
    out_channels = 1 if args.binary_mask else 2

    model = get_model(
        encoder=args.encoder,
        model_type=args.model_type,
        in_channels=len(args.incl_bands),
        output_channels=out_channels,
        pretrained=args.pretrained,
        freeze_encoder=args.freeze_encoder,
        weight_init=args.weight_init,
    )
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss() if args.binary_mask else nn.CrossEntropyLoss()

    opt_map = {
        "adam":  torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd":   torch.optim.SGD,
    }
    optimizer = opt_map[args.optimizer](model.parameters(), lr=args.lr)

    min_loss = np.inf
    epochs_no_improve = 0
    epoch_losses = []
    best_state_dict = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs} | ", end="")

        model.train()
        for images, target in train_loader:
            images, target = images.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(images), target)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, target in valid_loader:
                images, target = images.to(args.device), target.to(args.device)
                valid_loss += criterion(model(images), target).item()

        valid_loss /= len(valid_loader)
        epoch_losses.append(valid_loss)
        print(f"Validation Loss: {valid_loss:.5f}")

        if valid_loss < min_loss - args.early_stopping_min_delta:
            min_loss = valid_loss
            epochs_no_improve = 0
            m = model.module if isinstance(model, torch.nn.DataParallel) else model
            best_state_dict = copy.deepcopy(m.state_dict())
        else:
            epochs_no_improve += 1

        if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
            print("Early stopping triggered.")
            break

    return min_loss, epoch_losses, best_state_dict


# ---------------------------------------------------------
# LR sweep
# ---------------------------------------------------------

def _run_lr_sweep(train_loader, valid_loader, args):
    all_epoch_losses = {}
    best_loss, best_lr, best_state_dict = np.inf, None, None

    for lr in args.lr:
        print(f"\n--- LR: {lr} ---")
        run_args = copy.copy(args)
        run_args.lr = lr
        loss, epoch_losses, state_dict = train_model(train_loader, valid_loader, run_args)
        all_epoch_losses[lr] = epoch_losses
        if loss < best_loss:
            best_loss, best_lr, best_state_dict = loss, lr, state_dict

    torch.save(best_state_dict, os.path.join(args.save_path, args.model_name + ".pth"))

    config = _args_to_dict(args)
    config["epoch_losses"] = all_epoch_losses
    config["best_lr"] = best_lr
    config["best_loss"] = best_loss
    with open(os.path.join(args.save_path, args.model_name + ".json"), "w") as f:
        json.dump(config, f, indent=2)

    if len(args.lr) > 1:
        print(f"\nBest LR: {best_lr} (loss: {best_loss:.5f})")


# ---------------------------------------------------------
# Experiment mode (call from Python without CLI)
# ---------------------------------------------------------

def run_experiment(param_dict):
    """Run training by passing a dict instead of CLI args."""
    class Args:
        pass

    args = Args()
    for k, v in param_dict.items():
        setattr(args, k, v)

    if isinstance(args.incl_bands, str):
        args.incl_bands = np.array(eval(args.incl_bands)) - 1

    if isinstance(args.aug_contrast, str):
        args.aug_contrast = eval(args.aug_contrast)
    if isinstance(args.aug_brightness, str):
        args.aug_brightness = eval(args.aug_brightness)

    args.device = torch.device(args.device)

    if not isinstance(args.lr, list):
        args.lr = [args.lr]

    _validate_args(args)
    train_loader, valid_loader = load_data(args)
    _run_lr_sweep(train_loader, valid_loader, args)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _args_to_dict(args):
    d = vars(args).copy()
    d["device"] = str(d["device"])
    if hasattr(d.get("incl_bands"), "tolist"):
        d["incl_bands"] = d["incl_bands"].tolist()
    return d


# ---------------------------------------------------------
# Argument validation
# ---------------------------------------------------------

def _validate_args(args):
    os.makedirs(args.save_path, exist_ok=True)

    valid_model_types = ["unet", "r2_unet", "att_unet", "r2att_unet", "swed_unet"]
    if args.model_type not in valid_model_types:
        raise ValueError(f"model_type must be one of {valid_model_types}")

    valid_encoders = ["scratch", "resnet18", "resnet50", "resnet101"]
    if args.encoder not in valid_encoders:
        raise ValueError(f"encoder must be one of {valid_encoders}")

    valid_pretrained = ["none", "imagenet", "bigearthnet"]
    if args.pretrained not in valid_pretrained:
        raise ValueError(f"pretrained must be one of {valid_pretrained}")

    if args.pretrained != "none" and args.encoder == "scratch":
        raise ValueError("pretrained requires a ResNet encoder, not 'scratch'")

    if args.encoder != "scratch" and args.model_type in ("r2_unet", "r2att_unet", "swed_unet"):
        raise ValueError(
            f"model_type='{args.model_type}' is not compatible with a pretrained encoder. "
            f"Use 'unet' or 'att_unet'."
        )

    if not (0 < args.split < 1):
        raise ValueError("split must be between 0 and 1")

    if args.valid_path is not None and not os.path.isdir(args.valid_path):
        raise ValueError(f"valid_path does not exist: {args.valid_path}")

    valid_aug = ["none", "geometric", "gaussian_noise", "salt_pepper", "contrast", "combined"]
    if args.augmentation not in valid_aug:
        raise ValueError(f"augmentation must be one of {valid_aug}")


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Naming
    parser.add_argument("--model_name", type=str, required=True)

    # Architecture
    parser.add_argument("--model_type", type=str, default="unet",
                        choices=["unet", "r2_unet", "att_unet", "r2att_unet", "swed_unet"])
    parser.add_argument("--encoder", type=str, default="scratch",
                        choices=["scratch", "resnet18", "resnet50", "resnet101"])
    parser.add_argument("--pretrained", type=str, default="none",
                        choices=["none", "imagenet", "bigearthnet"])
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze ResNet encoder weights (conv1 always trainable)")
    parser.add_argument("--weight_init", type=str, default="normal",
                        choices=["normal", "xavier", "kaiming", "orthogonal"],
                        help="Weight initialisation for scratch encoder")

    # Input / output
    parser.add_argument("--satellite", type=str, required=True,
                        choices=["landsat", "sentinel", "gaofen1", "gaofen6"])
    parser.add_argument("--incl_bands", type=str, default="[1,2,3,4,5,6,7]",
                        help="1-indexed band positions to use as model input")
    parser.add_argument("--target_pos", type=int, default=-1,
                        help="Band index of the segmentation mask")
    parser.add_argument("--binary_mask", action="store_true",
                        help="Use BCEWithLogitsLoss + 1 output channel (default: CrossEntropy + 2)")

    # Optimiser
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", type=float, nargs="+", default=[0.001],
                        help="Learning rate(s). Pass multiple values to compare, e.g. --lr 0.01 0.001 0.0001")

    # Training loop
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--early_stopping", type=int, default=-1,
                        help="Patience in epochs; -1 disables early stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4,
                        help="Minimum loss improvement to reset patience counter")
    parser.add_argument("--seed", type=int, default=42)

    # Augmentation
    parser.add_argument("--augmentation", type=str, default="none",
                        choices=["none", "geometric", "gaussian_noise", "salt_pepper",
                                 "contrast", "combined"])
    parser.add_argument("--aug_noise_std", type=float, default=0.1,
                        help="Std dev for Gaussian noise augmentation")
    parser.add_argument("--aug_sp_prob", type=float, default=0.1,
                        help="Salt & pepper noise probability")
    parser.add_argument("--aug_contrast", type=str, default="[0.6,0.8,1.2,1.4]",
                        help="Contrast scaling factors (list)")
    parser.add_argument("--aug_brightness", type=str, default="[-0.1,0.1]",
                        help="Brightness offset range [min, max]")

    # Paths
    parser.add_argument("--train_path", type=str, default="../data/training/")
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="../models/")
    parser.add_argument("--finetune_from", type=str, default=None,
                        help="URL or local path to a pretrained .pth state dict")

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample", action="store_true",
                        help="Use only first 100 samples (for quick testing)")
    parser.add_argument("--note", type=str, default="")

    args = parser.parse_args()

    # Post-process list args
    args.incl_bands = np.array(eval(args.incl_bands)) - 1
    args.aug_contrast = eval(args.aug_contrast)
    args.aug_brightness = eval(args.aug_brightness)
    args.device = torch.device(args.device)

    print("\nTraining with args:\n", vars(args))

    _validate_args(args)
    train_loader, valid_loader = load_data(args)
    _run_lr_sweep(train_loader, valid_loader, args)


if __name__ == "__main__":
    main()
