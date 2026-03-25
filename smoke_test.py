"""
Smoke test — runs a small grid of training configurations for 1 epoch on 100 samples.
Use this locally before submitting a full run to HPC.

Usage:
    python smoke_test.py --train_path ../data/training/ --satellite landsat
    python smoke_test.py --train_path ../data/training/ --satellite sentinel --incl_bands "[1,2,3,4,5,6,7,8,9,10,11,12]"
"""

import sys
import os
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
from train import run_experiment, load_data
from network import get_model

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
# Configurations to test
# ---------------------------------------------------------

# Each entry is (label, overrides). Overrides are merged into base_config.
CONFIGS = [
    ("scratch / unet",          dict(model_type="unet",     encoder="scratch")),
    ("scratch / att_unet",      dict(model_type="att_unet", encoder="scratch")),
    ("scratch / r2_unet",       dict(model_type="r2_unet",  encoder="scratch")),
    ("resnet18 / unet / none",  dict(encoder="resnet18", model_type="unet",     pretrained="none")),
    ("resnet18 / att_unet / none", dict(encoder="resnet18", model_type="att_unet", pretrained="none")),
    ("resnet50 / unet / none",  dict(encoder="resnet50", model_type="unet",     pretrained="none")),
    ("resnet18 / unet / imagenet",  dict(encoder="resnet18", model_type="unet", pretrained="imagenet")),
    ("resnet50 / unet / imagenet",  dict(encoder="resnet50", model_type="unet", pretrained="imagenet")),
    ("resnet18 / unet / imagenet / frozen", dict(encoder="resnet18", model_type="unet",
                                                  pretrained="imagenet", freeze_encoder=True)),
    ("resnet18 / unet / geometric aug", dict(encoder="resnet18", model_type="unet",
                                              pretrained="none", augmentation="geometric")),
    ("resnet18 / unet / combined aug",  dict(encoder="resnet18", model_type="unet",
                                              pretrained="none", augmentation="combined")),
    ("resnet18 / unet / bigearthnet",         dict(encoder="resnet18", model_type="unet",
                                                    pretrained="bigearthnet")),
    ("resnet50 / unet / bigearthnet",         dict(encoder="resnet50", model_type="unet",
                                                    pretrained="bigearthnet")),
    ("resnet101 / unet / bigearthnet",        dict(encoder="resnet101", model_type="unet",
                                                    pretrained="bigearthnet")),
    ("resnet50 / att_unet / bigearthnet",     dict(encoder="resnet50", model_type="att_unet",
                                                    pretrained="bigearthnet")),
    ("resnet50 / unet / bigearthnet / frozen", dict(encoder="resnet50", model_type="unet",
                                                     pretrained="bigearthnet", freeze_encoder=True)),
]


# ---------------------------------------------------------
# Sense checks
# ---------------------------------------------------------

def _build_args(config):
    """Convert a config dict to an args object, mirroring run_experiment."""
    class Args:
        pass

    args = Args()
    for k, v in config.items():
        setattr(args, k, v)

    if isinstance(args.incl_bands, str):
        args.incl_bands = np.array(eval(args.incl_bands)) - 1
    else:
        args.incl_bands = np.array(args.incl_bands) - 1

    if isinstance(args.aug_contrast, str):
        args.aug_contrast = eval(args.aug_contrast)
    if isinstance(args.aug_brightness, str):
        args.aug_brightness = eval(args.aug_brightness)

    if not isinstance(args.lr, list):
        args.lr = [args.lr]

    args.device = torch.device(args.device)
    return args


def sense_check(config):
    """
    Instantiate model and data for one batch and run pre-training sanity checks.
    Prints a summary and raises RuntimeError on any critical failure.
    """
    args = _build_args(config)
    os.makedirs(args.save_path, exist_ok=True)

    # --- Data ---
    train_loader, _ = load_data(args)
    n_train = len(train_loader.dataset)
    images, targets = next(iter(train_loader))

    data_min  = images.min().item()
    data_max  = images.max().item()
    data_mean = images.mean().item()
    data_dtype = str(images.dtype)
    target_vals = sorted(set(targets.unique().cpu().long().tolist()))

    # --- Model ---
    out_channels = 1 if args.binary_mask else 2
    model = get_model(
        encoder=args.encoder,
        model_type=args.model_type,
        in_channels=len(args.incl_bands),
        output_channels=out_channels,
        pretrained=args.pretrained,
        freeze_encoder=args.freeze_encoder,
        weight_init=args.weight_init,
    ).to(args.device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Forward + loss + gradients ---
    images_d  = images.to(args.device)
    targets_d = targets.to(args.device)

    output = model(images_d)
    shape_match = output.shape == targets_d.shape

    criterion = nn.BCEWithLogitsLoss() if args.binary_mask else nn.CrossEntropyLoss()
    loss = criterion(output, targets_d)
    loss_val    = loss.item()
    loss_finite = np.isfinite(loss_val)

    loss.backward()
    has_grads = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters() if p.requires_grad
    )

    # --- Print summary ---
    W = 22
    range_ok = 0.0 <= data_min and data_max <= 1.0
    dtype_ok  = data_dtype == "torch.float32"

    print(f"    {'Training samples':{W}}: {n_train}")
    print(f"    {'Data range':{W}}: [{data_min:.4f}, {data_max:.4f}]  "
          f"mean={data_mean:.4f}  dtype={data_dtype}  "
          f"{'[OK]' if range_ok and dtype_ok else '[WARN]'}")
    print(f"    {'Target values':{W}}: {target_vals}")
    print(f"    {'Parameters':{W}}: {total_params:,} total  /  {trainable_params:,} trainable")
    print(f"    {'Output shape':{W}}: {tuple(output.shape)}  {'[OK]' if shape_match else '[FAIL]'}")
    print(f"    {'First-batch loss':{W}}: {loss_val:.5f}  {'[OK]' if loss_finite else '[FAIL]'}")
    print(f"    {'Gradient flow':{W}}: {'[OK]' if has_grads else '[FAIL]'}")

    # --- Raise on critical failures ---
    failures = []
    if not dtype_ok:
        failures.append(f"data dtype is {data_dtype}, expected torch.float32")
    if not shape_match:
        failures.append(f"output {tuple(output.shape)} != target {tuple(targets_d.shape)}")
    if not loss_finite:
        failures.append(f"loss is not finite ({loss_val})")
    if not has_grads:
        failures.append("no gradients flowing")

    if failures:
        raise RuntimeError("Sense check failed: " + "; ".join(failures))


# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------

def run_smoke_test(args):
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Train path: {args.train_path}")
    print(f"Satellite: {args.satellite}")
    print(f"Bands: {args.incl_bands}")
    print(f"Running {len(CONFIGS)} configurations...\n")
    print("-" * 65)

    base_config = dict(
        model_name       = "smoke_test",
        satellite        = args.satellite,
        incl_bands       = args.incl_bands,
        target_pos       = -1,
        binary_mask      = False,
        encoder          = "scratch",
        model_type       = "unet",
        pretrained       = "none",
        freeze_encoder   = False,
        weight_init      = "normal",
        optimizer        = "adam",
        lr               = 0.001,
        batch_size       = 8,
        epochs           = 1,
        split            = 0.8,
        early_stopping   = -1,
        seed             = 42,
        augmentation     = "none",
        aug_noise_std    = 0.1,
        aug_sp_prob      = 0.1,
        aug_contrast     = [0.6, 0.8, 1.2, 1.4],
        aug_brightness   = [-0.1, 0.1],
        train_path       = args.train_path,
        valid_path       = None,
        save_path        = "/tmp/smoke_test_models/",
        finetune_from    = None,
        sample           = True,   # first 100 samples only
        device           = device,
        note             = "smoke_test",
    )

    results = []

    for label, overrides in CONFIGS:
        config = {**base_config, **overrides}
        print(f"  {label}")
        t0 = time.time()
        try:
            sense_check(config)
            run_experiment(config)
            elapsed = time.time() - t0
            print(f"  PASSED  ({elapsed:.1f}s)")
            results.append((label, True, None))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED  ({elapsed:.1f}s)")
            print(f"    {type(e).__name__}: {e}")
            results.append((label, False, str(e)))

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print("-" * 65)
    print(f"\n{passed}/{len(results)} configurations passed.")

    if failed:
        print("\nFailed configurations:")
        for label, ok, err in results:
            if not ok:
                print(f"  - {label}: {err}")
        sys.exit(1)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for training configurations")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to directory of training .npy files")
    parser.add_argument("--satellite", type=str, required=True,
                        choices=["landsat", "sentinel", "gaofen1", "gaofen6"])
    parser.add_argument("--incl_bands", type=str, default="[1,2,3,4,5,6,7]",
                        help="1-indexed band positions to use as model input")
    args = parser.parse_args()
    run_smoke_test(args)
