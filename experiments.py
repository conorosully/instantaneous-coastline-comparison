# Experiment runner for LICS coastline segmentation
# Conor O'Sullivan
#
# Usage:
#   python experiments.py --train_path ../data/LICS/train --finetune_path ../data/LICS/finetune --save_path ../models/
#   python experiments.py --train_path ... --finetune_path ... --save_path ... --experiment 2

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from train import run_experiment


# ---------------------------------------------------------
# Device detection
# ---------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------
# Base config
# ---------------------------------------------------------

def _base(train_path, save_path):
    return dict(
        satellite        = "landsat",
        incl_bands       = "[1,2,3,4,5,6,7]",
        target_pos       = -1,
        binary_mask      = False,
        encoder          = "scratch",
        model_type       = "unet",
        pretrained       = "none",
        freeze_encoder   = False,
        weight_init      = "normal",
        optimizer        = "adam",
        lr               = [0.1, 0.01, 0.001],
        batch_size       = 32,
        epochs           = 50,
        split            = 0.9,
        early_stopping   = 10,
        seed             = 42,
        augmentation     = "none",
        aug_noise_std    = 0.1,
        aug_sp_prob      = 0.1,
        aug_contrast     = [0.6, 0.8, 1.2, 1.4],
        aug_brightness   = [-0.1, 0.1],
        train_path       = train_path,
        valid_path       = None,
        save_path        = save_path,
        finetune_from    = None,
        sample           = False,
        device           = get_device(),
        note             = "",
    )


# ---------------------------------------------------------
# Experiment 1 — Architecture comparison
# ---------------------------------------------------------

def exp1_architectures(train_path, save_path):
    """
    All scratch architectures × all optimizers × both loss functions.
    Trained on the LICS training set.
    """
    print("\n" + "=" * 65)
    print("Experiment 1: Architecture Comparison")
    print("=" * 65)

    architectures = ["unet", "r2_unet", "att_unet", "r2att_unet"]
    optimizers    = ["adam", "adamw", "sgd"]

    for arch in architectures:
        for opt in optimizers:
            model_name = f"LICS_{arch}_{opt}"
            print(f"\n  {model_name}")
            run_experiment({
                **_base(train_path, save_path),
                "model_name": model_name,
                "model_type": arch,
                "optimizer":  opt,
            })


# ---------------------------------------------------------
# Experiment 2 — Augmentation comparison
# ---------------------------------------------------------

def exp2_augmentations(finetune_path, save_path):
    """
    UNet × all augmentation types × on adam × both loss functions.
    Trained on the LICS finetuning set.
    """
    print("\n" + "=" * 65)
    print("Experiment 2: Augmentation Comparison")
    print("=" * 65)

    augmentations = ["none", "geometric", "gaussian_noise", "salt_pepper", "contrast", "combined"]
    optimizers    = ["adam"]

    for aug in augmentations:
        for opt in optimizers:
            model_name = f"LICS_{aug}_{opt}"
            print(f"\n  {model_name}")
            run_experiment({
                **_base(finetune_path, save_path),
                "model_name":   model_name,
                "augmentation": aug,
                "optimizer":    opt,
            })


# ---------------------------------------------------------
# Experiment 3 — Pretrained encoder comparison
# ---------------------------------------------------------

def exp3_encoders(train_path, save_path):
    """
    UNet × ResNet encoders × pretrained weights. Optimizer: Adam, Loss: CrossEntropy.
    Trained on the LICS training set.
    """
    print("\n" + "=" * 65)
    print("Experiment 3: Pretrained Encoder Comparison")
    print("=" * 65)

    encoders   = ["resnet18", "resnet50", "resnet101"]
    pretrained = ["imagenet", "bigearthnet"]

    for encoder in encoders:
        for pretrain in pretrained:
            model_name = f"LICS_{encoder}_{pretrain}_unfrozen"
            print(f"\n  {model_name}")
            run_experiment({
                **_base(train_path, save_path),
                "model_name":    model_name,
                "model_type":    "unet",
                "encoder":       encoder,
                "pretrained":    pretrain,
                "optimizer":     "adam",
                "binary_mask":   False,
                "freeze_encoder": False,
            })


# ---------------------------------------------------------
# Experiment 4 — Frozen pretrained encoder comparison
# ---------------------------------------------------------

def exp4_frozen_encoders(train_path, save_path):
    """
    Same as Experiment 3 but with frozen encoder weights.
    UNet × ResNet encoders × pretrained weights. Optimizer: Adam.
    Trained on the LICS training set.
    """
    print("\n" + "=" * 65)
    print("Experiment 4: Frozen Pretrained Encoder Comparison")
    print("=" * 65)

    encoders   = ["resnet18", "resnet50", "resnet101"]
    pretrained = ["imagenet", "bigearthnet"]

    for encoder in encoders:
        for pretrain in pretrained:
            model_name = f"LICS_{encoder}_{pretrain}_frozen"
            print(f"\n  {model_name}")
            run_experiment({
                **_base(train_path, save_path),
                "model_name":    model_name,
                "model_type":    "unet",
                "encoder":       encoder,
                "pretrained":    pretrain,
                "optimizer":     "adam",
                "binary_mask":   False,
                "freeze_encoder": True,
            })


# ---------------------------------------------------------
# Experiment 5 — Dataset comparison
# ---------------------------------------------------------

def exp5_datasets(scratch_path, save_path):
    """
    UNet + Adam across SWED, SANet, and TCNet datasets.
    Trained on each dataset's training set.
    """
    print("\n" + "=" * 65)
    print("Experiment 5: Dataset Comparison")
    print("=" * 65)

    datasets = [
        ("SWED",             os.path.join(scratch_path, "SWED",             "train"),
            {"satellite": "sentinel", "batch_size": 8}),
        ("SANet_processed",  os.path.join(scratch_path, "SANet_processed",  "train"),
            {"incl_bands": "[1,2,3,4]", "target_pos": -1, "satellite": "gaofen1"}),
        ("TCUNet_processed", os.path.join(scratch_path, "TCUNet_processed", "train"),
            {"incl_bands": "[1,2,3,4,5,6,7,8]", "target_pos": -1, "satellite": "gaofen6"}),
    ]

    for dataset, dataset_train_path, overrides in datasets:
        model_name = f"{dataset}_unet_adam"
        print(f"\n  {model_name}")
        run_experiment({
            **_base(dataset_train_path, save_path),
            "model_name": model_name,
            "model_type": "unet",
            "optimizer":  "adam",
            **overrides,
        })


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run LICS segmentation experiments")
    parser.add_argument("--train_path",    type=str, required=True,
                        help="Path to LICS training data (.npy files)")
    parser.add_argument("--finetune_path", type=str, default=None,
                        help="Path to LICS finetuning data (required for experiment 2)")
    parser.add_argument("--scratch_path",  type=str, default=None,
                        help="Path to scratch directory containing SWED, SANet_processed, TCUNet_processed (required for experiment 5)")
    parser.add_argument("--save_path",     type=str, required=True,
                        help="Directory to save models and configs")
    parser.add_argument("--experiment",    type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Run a specific experiment (1-5). Omit to run all.")
    args = parser.parse_args()

    if args.experiment in (None, 2) and args.finetune_path is None:
        parser.error("--finetune_path is required for experiment 2")

    if args.experiment in (None, 5) and args.scratch_path is None:
        parser.error("--scratch_path is required for experiment 5")

    os.makedirs(args.save_path, exist_ok=True)

    run_all = args.experiment is None

    if run_all or args.experiment == 1:
        exp1_architectures(args.train_path, args.save_path)

    if run_all or args.experiment == 2:
        exp2_augmentations(args.finetune_path, args.save_path)

    if run_all or args.experiment == 3:
        exp3_encoders(args.train_path, args.save_path)

    if run_all or args.experiment == 4:
        exp4_frozen_encoders(args.train_path, args.save_path)

    if run_all or args.experiment == 5:
        exp5_datasets(args.scratch_path, args.save_path)


if __name__ == "__main__":
    main()
