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
from network import load_model


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
# Experiment 5 — Dataset comparison (sub-experiments 5a–5d)
#
# Each sub-experiment sweeps:
#   optimizers   : adam, adamw, sgd
#   augmentations: none, geometric
#   lr           : [0.1, 0.01, 0.001, 0.0001]
#
# Model naming: {dataset}_unet_{aug}_{opt}
# ---------------------------------------------------------

def _exp5_sweep(dataset, train_path, save_path, overrides):
    optimizers    = ["adam", "adamw", "sgd"]
    augmentations = ["none", "geometric"]

    for aug in augmentations:
        for opt in optimizers:
            model_name = f"{dataset}_unet_{aug}_{opt}"
            print(f"\n  {model_name}")
            run_experiment({
                **_base(train_path, save_path),
                "model_name":   model_name,
                "model_type":   "unet",
                "optimizer":    opt,
                "lr":           [0.1, 0.01, 0.001, 0.0001],
                "augmentation": aug,
                **overrides,
            })


def exp5a_lics(train_path, save_path):
    """Experiment 5a: UNet hyperparameter sweep on LICS (Landsat)."""
    print("\n" + "=" * 65)
    print("Experiment 5a: Dataset Comparison — LICS")
    print("=" * 65)
    _exp5_sweep("LICS", train_path, save_path, overrides={})


def exp5b_swed(scratch_path, save_path):
    """Experiment 5b: UNet hyperparameter sweep on SWED (Sentinel-2)."""
    print("\n" + "=" * 65)
    print("Experiment 5b: Dataset Comparison — SWED")
    print("=" * 65)
    _exp5_sweep(
        "SWED",
        os.path.join(scratch_path, "SWED", "train"),
        save_path,
        overrides={"incl_bands": "[1,2,3,4,5,6,7,8,9,10,11,12]", "satellite": "sentinel", "batch_size": 8},
    )


def exp5c_sanet(scratch_path, save_path):
    """Experiment 5c: UNet hyperparameter sweep on SANet_processed (Gaofen-1)."""
    print("\n" + "=" * 65)
    print("Experiment 5c: Dataset Comparison — SANet_processed")
    print("=" * 65)
    _exp5_sweep(
        "SANet_processed",
        os.path.join(scratch_path, "SANet_processed", "train"),
        save_path,
        overrides={"incl_bands": "[1,2,3,4]", "target_pos": -1, "satellite": "gaofen1"},
    )


def exp5d_tcunet(scratch_path, save_path):
    """Experiment 5d: UNet hyperparameter sweep on TCUNet_processed (Gaofen-6)."""
    print("\n" + "=" * 65)
    print("Experiment 5d: Dataset Comparison — TCUNet_processed")
    print("=" * 65)
    _exp5_sweep(
        "TCUNet_processed",
        os.path.join(scratch_path, "TCUNet_processed", "train"),
        save_path,
        overrides={"incl_bands": "[1,2,3,4,5,6,7,8]", "target_pos": -1, "satellite": "gaofen6", "batch_size": 8},
    )


# ---------------------------------------------------------
# Evaluate all saved models → CSV
# ---------------------------------------------------------

def _dataset_name(model_name):
    for prefix in ("SWED", "SANet_processed", "TCUNet_processed"):
        if model_name.startswith(prefix):
            return prefix
    return "LICS"


def _experiment_number(config):
    name    = config["model_name"]
    encoder = config["encoder"]

    if not name.startswith("LICS"):
        return 5

    if encoder != "scratch":
        return 4 if config["freeze_encoder"] else 3

    architectures = {"unet", "r2_unet", "att_unet", "r2att_unet"}
    stem = name[len("LICS_"):]
    for opt in ("adam", "adamw", "sgd"):
        if stem.endswith(f"_{opt}"):
            stem = stem[:-(len(opt) + 1)]
            break
    return 1 if stem in architectures else 2


def evaluate_all(models_dir, test_paths, output_csv):
    """
    Run evaluation for every model in models_dir and write results to a CSV.

    models_dir : directory containing .pth / .json pairs
    test_paths : dict mapping dataset name to its test directory
                 e.g. {"LICS": "data/LICS/test",
                        "SWED": "data/SWED/test",
                        "SANet_processed": "data/SANet_processed/test",
                        "TCUNet_processed": "data/TCUNet_processed/test"}
    output_csv : path to write the results CSV
    """
    import csv
    import glob
    import types
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset import TrainDataset
    from evaluation import eval_metrics

    device = get_device()

    json_files = sorted(glob.glob(os.path.join(models_dir, "*.json")))

    fieldnames = [
        "model_name", "experiment",
        "model_type", "encoder", "pretrained", "freeze_encoder",
        "augmentation", "optimizer", "best_lr", "best_loss", "epochs_trained",
        "satellite", "n_params",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mse", "fom",
    ]

    rows = []
    for json_path in json_files:
        import json
        with open(json_path) as f:
            config = json.load(f)

        model_name  = config["model_name"]
        dataset     = _dataset_name(model_name)
        test_dir    = test_paths.get(dataset)

        if test_dir is None:
            print(f"  Skipping {model_name} — no test path for dataset '{dataset}'")
            continue

        test_files = glob.glob(os.path.join(test_dir, "*.npy"))
        if not test_files:
            print(f"  Skipping {model_name} — no .npy files in {test_dir}")
            continue

        print(f"  Evaluating {model_name} ({len(test_files)} test images)...")

        args = types.SimpleNamespace(
            target_pos   = config["target_pos"],
            incl_bands   = config["incl_bands"],
            satellite    = config["satellite"],
            binary_mask  = config.get("binary_mask", False),
            augmentation = "none",
            aug_noise_std  = 0.1,
            aug_sp_prob    = 0.1,
            aug_contrast   = [0.6, 0.8, 1.2, 1.4],
            aug_brightness = [-0.1, 0.1],
        )

        if dataset == "LICS":
            args.target_pos = -2  # LICS has edge mask in last channel

        loader = DataLoader(TrainDataset(test_files, args), batch_size=16)
        model, _ = load_model(model_name, models_dir, device=device)
        n_params = sum(p.numel() for p in model.parameters())

        targets_list, preds_list = [], []
        with torch.no_grad():
            for images, targets in loader:
                images  = images.to(device)
                outputs = model(images)

                if args.binary_mask:
                    pred = (torch.sigmoid(outputs).squeeze(1) > 0.5).cpu().numpy().astype(int)
                    tgt  = (targets.squeeze(1) > 0.5).numpy().astype(int)
                else:
                    pred = outputs.argmax(dim=1).cpu().numpy()
                    tgt  = targets[:, 1].numpy().astype(int)

                for i in range(len(pred)):
                    preds_list.append(pred[i])
                    targets_list.append(tgt[i])

        metrics, _ = eval_metrics(targets_list, preds_list)
        epochs_trained = sum(len(v) for v in config["epoch_losses"].values())

        rows.append({
            "model_name":       model_name,
            "experiment":       _experiment_number(config),
            "model_type":       config["model_type"],
            "encoder":          config["encoder"],
            "pretrained":       config.get("pretrained", "none"),
            "freeze_encoder":   config.get("freeze_encoder", False),
            "augmentation":     config.get("augmentation", "none"),
            "optimizer":        config["optimizer"],
            "best_lr":          config["best_lr"],
            "best_loss":        config["best_loss"],
            "epochs_trained":   epochs_trained,
            "satellite":        config["satellite"],
            "n_params":         n_params,
            **metrics,
        })

        print(f"F1={metrics['f1']:.4f}  FoM={metrics['fom']:.4f}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    

    print(f"\nResults written to {output_csv} ({len(rows)} models evaluated)")



# ---------------------------------------------------------
# Spectral index baseline evaluation → CSV
# ---------------------------------------------------------

# Satellite and target-band metadata for each known dataset
_DATASET_META = {
    "LICS":             {"satellite": "landsat",  "target_pos": -2},
    "SWED":             {"satellite": "sentinel",  "target_pos": -1},
    "SANet_processed":  {"satellite": "gaofen1",   "target_pos": -1},
    "TCUNet_processed": {"satellite": "gaofen6",   "target_pos": -1},
}


def evaluate_index_method(test_paths, output_csv, index="MNDWI", threshold="otsu"):
    """
    Evaluate a spectral index + threshold baseline across datasets and write a CSV.

    test_paths : dict mapping dataset name to its test directory of .npy files
                 e.g. {"LICS": "/data/LICS/test", "SWED": "data/SWED/test", ...}
    output_csv : path to write the results CSV
    index      : any spyndex-supported index name (default "MNDWI")
    threshold  : "otsu" for Otsu's method, or a float for a fixed cutoff
    """
    import csv
    import glob
    import numpy as np
    from utils import predict_index
    from evaluation import eval_metrics

    fieldnames = [
        "dataset", "satellite", "index", "threshold",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mse", "fom",
    ]

    rows = []
    for dataset, test_dir in sorted(test_paths.items()):
        meta = _DATASET_META.get(dataset)
        if meta is None:
            print(f"  Skipping {dataset} — no metadata entry in _DATASET_META")
            continue

        test_files = glob.glob(os.path.join(test_dir, "*.npy"))
        if not test_files:
            print(f"  Skipping {dataset} — no .npy files in {test_dir}")
            continue

        satellite  = meta["satellite"]
        target_pos = meta["target_pos"]
        print(f"  Evaluating index={index} on {dataset} ({len(test_files)} images)...")

        targets_list, preds_list = [], []
        for path in test_files:
            arr = np.load(path)
            tgt = arr[:, :, target_pos].astype(int)
            tgt[tgt == -1] = 0

            # get_index selects the relevant spectral bands internally,
            # so the full array can be passed directly
            pred = predict_index(arr, satellite=satellite, index=index, threshold=threshold)

            targets_list.append(tgt)
            preds_list.append(pred)

        metrics, _ = eval_metrics(targets_list, preds_list)

        rows.append({
            "dataset":   dataset,
            "satellite": satellite,
            "index":     index,
            "threshold": threshold,
            **metrics,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_csv} ({len(rows)} datasets evaluated)")


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run LICS segmentation experiments")
    parser.add_argument("--train_path",    type=str, default=None,
                        help="Path to LICS training data (.npy files)")
    parser.add_argument("--finetune_path", type=str, default=None,
                        help="Path to LICS finetuning data (required for experiment 2)")
    parser.add_argument("--scratch_path",  type=str, default=None,
                        help="Path to scratch directory containing SWED, SANet_processed, TCUNet_processed (required for experiment 5)")
    parser.add_argument("--save_path",     type=str, default=None,
                        help="Directory to save models and configs")
    parser.add_argument("--experiment",    type=str,
                        choices=["1", "2", "3", "4", "5", "5a", "5b", "5c", "5d"],
                        default=None,
                        help="Run a specific experiment. 5=all datasets, 5a=LICS, 5b=SWED, 5c=SANet, 5d=TCUNet. Omit to run all.")

    # Evaluation mode
    parser.add_argument("--evaluate",      action="store_true",
                        help="Evaluate all models in --models_dir and write a CSV")
    parser.add_argument("--models_dir",    type=str, default=None,
                        help="Directory of saved .pth/.json models (required for --evaluate)")
    parser.add_argument("--lics_test",     type=str, default=None,
                        help="Path to LICS test set (.npy files)")
    parser.add_argument("--swed_test",     type=str, default=None,
                        help="Path to SWED test set (.npy files)")
    parser.add_argument("--sanet_test",    type=str, default=None,
                        help="Path to SANet_processed test set (.npy files)")
    parser.add_argument("--tcunet_test",   type=str, default=None,
                        help="Path to TCUNet_processed test set (.npy files)")
    parser.add_argument("--output_csv",    type=str, default="results.csv",
                        help="Output CSV path (default: results.csv)")

    args = parser.parse_args()

    if args.evaluate:
        if args.models_dir is None:
            parser.error("--models_dir is required for --evaluate")
        test_paths = {k: v for k, v in {
            "LICS":             args.lics_test,
            "SWED":             args.swed_test,
            "SANet_processed":  args.sanet_test,
            "TCUNet_processed": args.tcunet_test,
        }.items() if v is not None}
        evaluate_all(args.models_dir, test_paths, args.output_csv)
        return

    if args.save_path is None:
        parser.error("--save_path is required for training")

    needs_train  = args.experiment in (None, "1", "2", "3", "4", "5", "5a")
    needs_scratch = args.experiment in (None, "5", "5b", "5c", "5d")

    if needs_train and args.train_path is None:
        parser.error("--train_path is required for this experiment")
    if args.experiment in (None, "2") and args.finetune_path is None:
        parser.error("--finetune_path is required for experiment 2")
    if needs_scratch and args.scratch_path is None:
        parser.error("--scratch_path is required for this experiment")

    os.makedirs(args.save_path, exist_ok=True)

    run_all = args.experiment is None
    exp     = args.experiment

    if run_all or exp == "1":
        exp1_architectures(args.train_path, args.save_path)

    if run_all or exp == "2":
        exp2_augmentations(args.finetune_path, args.save_path)

    if run_all or exp == "3":
        exp3_encoders(args.train_path, args.save_path)

    if run_all or exp == "4":
        exp4_frozen_encoders(args.train_path, args.save_path)

    if run_all or exp in ("5", "5a"):
        exp5a_lics(args.train_path, args.save_path)

    if run_all or exp in ("5", "5b"):
        exp5b_swed(args.scratch_path, args.save_path)

    if run_all or exp in ("5", "5c"):
        exp5c_sanet(args.scratch_path, args.save_path)

    if run_all or exp in ("5", "5d"):
        exp5d_tcunet(args.scratch_path, args.save_path)


if __name__ == "__main__":
    main()
