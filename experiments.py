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
        batch_size       = 8,
        epochs           = 100,
        split            = 0.9,
        early_stopping   = 20,
        early_stopping_min_delta = 1e-4,
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
# Experiment 2 — Architecture comparison
# ---------------------------------------------------------

_EXP2_ARCHITECTURES = ("unet", "r2_unet", "att_unet", "r2att_unet", "swed_unet")
_EXP2_DATASETS      = ("LICS", "SWED", "SANet_processed", "TCUNet_processed")


def _exp2_dataset_config(dataset, train_path, scratch_path):
    """Return dataset-specific config overrides for experiment 2."""
    if dataset == "LICS":
        return {
            "train_path":     train_path,
            "satellite":      "landsat",
            "incl_bands":     "[1,2,3,4,5,6,7]",
            "early_stopping": 10,
        }
    if dataset == "SWED":
        return {
            "train_path":     os.path.join(scratch_path, "SWED", "train"),
            "satellite":      "sentinel",
            "incl_bands":     "[1,2,3,4,5,6,7,8,9,10,11,12]",
            "early_stopping": 10,
        }
    if dataset == "SANet_processed":
        return {
            "train_path":     os.path.join(scratch_path, "SANet_processed", "train"),
            "valid_path":     os.path.join(scratch_path, "SANet_processed", "valid"),
            "satellite":      "gaofen1",
            "incl_bands":     "[1,2,3,4]",
            "early_stopping": 20,
        }
    if dataset == "TCUNet_processed":
        return {
            "train_path":     os.path.join(scratch_path, "TCUNet_processed", "train"),
            "satellite":      "gaofen6",
            "incl_bands":     "[1,2,3,4,5,6,7,8]",
            "early_stopping": 20,
            "batch_size":     4,   # Gaofen-6 tiles are large; 8 OOMs even for plain UNet
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def exp2_architectures(train_path, scratch_path, save_path, dataset=None, models=None):
    """
    Scratch architectures × adam/sgd × datasets.

    dataset : one of _EXP2_DATASETS, or None to run all four.
    models  : list of architectures to run, or None to run all five.
    """
    print("\n" + "=" * 65)
    print("Experiment 2: Architecture Comparison")
    if dataset:
        print(f"  dataset  : {dataset}")
    if models:
        print(f"  models   : {models}")
    print("=" * 65)

    datasets      = [dataset] if dataset else list(_EXP2_DATASETS)
    architectures = list(models) if models else list(_EXP2_ARCHITECTURES)
    optimizers    = ["adam", "sgd"]

    for ds in datasets:
        ds_cfg = _exp2_dataset_config(ds, train_path, scratch_path)
        for arch in architectures:
            for opt in optimizers:
                model_name = f"{ds}_{arch}_{opt}"
                print(f"\n  {model_name}")
                run_experiment({
                    **_base(ds_cfg["train_path"], save_path),
                    **ds_cfg,
                    "model_name":     model_name,
                    "model_type":     arch,
                    "optimizer":      opt,
                    "lr":             [0.1, 0.01, 0.001],
                    "experiment_tag": 2,
                })


# ---------------------------------------------------------
# Experiment 3 — Augmentation comparison
# ---------------------------------------------------------

def exp3_augmentations(dataset, finetune_path, save_path, overrides=None):
    """
    UNet × all augmentation types × all optimizers on a given dataset.
    dataset   : dataset name prefix for model naming (e.g. "LICS", "SWED")
    overrides : optional dict of config overrides (e.g. satellite, incl_bands)
    """
    print("\n" + "=" * 65)
    print(f"Experiment 3: Augmentation Comparison — {dataset}")
    print("=" * 65)

    augmentations = ["none", "geometric", "gaussian_noise", "salt_pepper", "contrast", "combined"]
    optimizers    = ["adam", "adamw", "sgd"]

    for aug in augmentations:
        for opt in optimizers:
            model_name = f"{dataset}_{aug}_{opt}"
            print(f"\n  {model_name}")
            run_experiment({
                **_base(finetune_path, save_path),
                "model_name":     model_name,
                "augmentation":   aug,
                "optimizer":      opt,
                "lr":             [0.1, 0.01, 0.001, 0.0001],
                "split":          0.8,
                "experiment_tag": 3,
                **(overrides or {}),
            })


# ---------------------------------------------------------
# Experiment 1 — Multi-dataset hyperparameter sweep
#
# Sweep: optimizers (adam, adamw, sgd) × lr candidates
#        (best lr picked automatically by _run_lr_sweep)
# Datasets: LICS (Landsat), SWED (Sentinel-2),
#           SANet_processed (Gaofen-1), TCUNet_processed (Gaofen-6)
#
# Model naming: {dataset}_unet_{opt}
# ---------------------------------------------------------

def _exp1_sweep(dataset, train_path, save_path, overrides):
    optimizers = ["adam", "adamw", "sgd"]

    for opt in optimizers:
        model_name = f"{dataset}_unet_{opt}"
        print(f"\n  {model_name}")
        run_experiment({
            **_base(train_path, save_path),
            "model_name":     model_name,
            "model_type":     "unet",
            "optimizer":      opt,
            "lr":             [0.1, 0.01, 0.001, 0.0001],
            "experiment_tag": 1,
            **overrides,
        })


_EXP1_DATASETS = ("LICS", "SWED", "SANet_processed", "TCUNet_processed")


def exp1_datasets(train_path, scratch_path, save_path, dataset=None):
    """
    Experiment 1: UNet hyperparameter sweep across optical datasets.

    dataset : one of _EXP1_DATASETS to run a single dataset, or None to run all.
    """
    print("\n" + "=" * 65)
    print("Experiment 1: Multi-Dataset Hyperparameter Sweep")
    if dataset:
        print(f"  (dataset filter: {dataset})")
    print("=" * 65)

    run_all = dataset is None

    if run_all or dataset == "LICS":
        _exp1_sweep("LICS", train_path, save_path, overrides={"early_stopping": 10})

    if run_all or dataset == "SWED":
        _exp1_sweep(
            "SWED",
            os.path.join(scratch_path, "SWED", "train"),
            save_path,
            overrides={"incl_bands": "[1,2,3,4,5,6,7,8,9,10,11,12]", "satellite": "sentinel",
                       "early_stopping": 10},
        )

    if run_all or dataset == "SANet_processed":
        _exp1_sweep(
            "SANet_processed",
            os.path.join(scratch_path, "SANet_processed", "train"),
            save_path,
            overrides={
                "incl_bands": "[1,2,3,4]", "target_pos": -1, "satellite": "gaofen1",
                "valid_path": os.path.join(scratch_path, "SANet_processed", "valid"),
            },
        )

    if run_all or dataset == "TCUNet_processed":
        _exp1_sweep(
            "TCUNet_processed",
            os.path.join(scratch_path, "TCUNet_processed", "train"),
            save_path,
            overrides={"incl_bands": "[1,2,3,4,5,6,7,8]", "target_pos": -1, "satellite": "gaofen6"},
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
    name = config["model_name"]

    if config.get("experiment_tag") is not None:
        return config["experiment_tag"]

    if not name.startswith("LICS"):
        return 1

    architectures = {"unet", "r2_unet", "att_unet", "r2att_unet", "swed_unet"}
    stem = name[len("LICS_"):]
    for opt in ("adam", "adamw", "sgd"):
        if stem.endswith(f"_{opt}"):
            stem = stem[:-(len(opt) + 1)]
            break
    return 2 if stem in architectures else 3


def evaluate_all(models_dir, test_paths, output_csv):
    """
    Run evaluation for every model in models_dir and write results to a CSV.

    models_dir : parent directory containing per-experiment subdirectories
                 (exp1/, exp2/, exp3/, …).
    test_paths : dict mapping dataset name to its test directory
                 e.g. {"LICS": "data/LICS/test",
                        "SWED": "data/SWED/test",
                        "SANet_processed": "data/SANet_processed/test",
                        "TCUNet_processed": "data/TCUNet_processed/test"}
    output_csv : path to write the results CSV
    """
    import csv
    import json
    import glob
    import types
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset import TrainDataset
    from evaluation import eval_metrics

    device = get_device()

    # Collect (json_path, model_dir, exp_num) from all exp*/ subdirectories,
    # sorted so results are grouped by experiment then model name.
    exp_dirs = sorted(
        e.path for e in os.scandir(models_dir)
        if e.is_dir() and e.name.startswith("exp") and e.name[3:].isdigit()
    )
    entries = [
        (json_path, exp_dir, int(os.path.basename(exp_dir)[3:]))
        for exp_dir in exp_dirs
        for json_path in sorted(glob.glob(os.path.join(exp_dir, "*.json")))
    ]

    fieldnames = [
        "model_name", "experiment",
        "model_type", "encoder", "pretrained", "freeze_encoder",
        "augmentation", "optimizer", "best_lr", "best_loss", "epochs_trained",
        "satellite", "n_params",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mse", "fom",
    ]

    rows = []
    for json_path, model_dir, exp_num in entries:
        with open(json_path) as f:
            config = json.load(f)

        model_name = config["model_name"]
        dataset    = _dataset_name(model_name)
        test_dir   = test_paths.get(dataset)

        if test_dir is None:
            print(f"  Skipping {model_name} — no test path for dataset '{dataset}'")
            continue

        test_files = glob.glob(os.path.join(test_dir, "*.npy"))
        if not test_files:
            print(f"  Skipping {model_name} — no .npy files in {test_dir}")
            continue

        print(f"  [exp{exp_num}] Evaluating {model_name} ({len(test_files)} test images)...")

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
        model, _ = load_model(model_name, model_dir, device=device)
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
            "experiment":       exp_num,
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
    parser.add_argument("--finetune_path",      type=str, default=None,
                        help="Path to LICS finetuning data (experiment 3)")
    parser.add_argument("--swed_finetune_path", type=str, default=None,
                        help="Path to SWED finetuning data (experiment 3 SWED)")
    parser.add_argument("--scratch_path",  type=str, default=None,
                        help="Path to scratch directory containing SWED, SANet_processed, TCUNet_processed (required for experiment 1)")
    parser.add_argument("--save_path",     type=str, default=None,
                        help="Directory to save models and configs")
    parser.add_argument("--experiment",    type=str,
                        choices=["1", "2", "3"],
                        default=None,
                        help="Run a specific experiment. Omit to run all.")
    parser.add_argument("--exp1_dataset",  type=str,
                        choices=list(_EXP1_DATASETS),
                        default=None,
                        help="(Experiment 1 only) Run a single dataset. Omit to run all four.")
    parser.add_argument("--exp2_dataset",   type=str,
                        choices=list(_EXP2_DATASETS),
                        default=None,
                        help="(Experiment 2 only) Run a single dataset. Omit to run all four.")
    exp2_model_group = parser.add_mutually_exclusive_group()
    exp2_model_group.add_argument("--exp2_models",    type=str, nargs="+",
                        choices=list(_EXP2_ARCHITECTURES),
                        default=None,
                        help="(Experiment 2 only) List of architectures to run, e.g. --exp2_models unet swed_unet att_unet")
    exp2_model_group.add_argument("--exp2_all_models", action="store_true",
                        help="(Experiment 2 only) Run all architectures (default if neither flag is given)")

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

    needs_train   = args.experiment in (None, "2")
    needs_scratch = args.experiment in (None, "1", "2")

    if needs_train and args.train_path is None:
        parser.error("--train_path is required for this experiment")
    if args.experiment in (None, "3") and args.finetune_path is None and args.swed_finetune_path is None:
        parser.error("--finetune_path and/or --swed_finetune_path is required for experiment 3")
    if needs_scratch and args.scratch_path is None:
        parser.error("--scratch_path is required for this experiment")

    os.makedirs(args.save_path, exist_ok=True)

    run_all = args.experiment is None
    exp     = args.experiment

    if run_all or exp == "1":
        exp1_datasets(args.train_path, args.scratch_path, args.save_path,
                      dataset=args.exp1_dataset)

    if run_all or exp == "2":
        exp2_models = args.exp2_models if not args.exp2_all_models else None
        exp2_architectures(args.train_path, args.scratch_path, args.save_path,
                           dataset=args.exp2_dataset,
                           models=exp2_models)

    if run_all or exp == "3":
        if args.finetune_path:
            exp3_augmentations("LICS", args.finetune_path, args.save_path)
        if args.swed_finetune_path:
            exp3_augmentations("SWED", args.swed_finetune_path, args.save_path,
                               overrides={"satellite": "sentinel",
                                          "incl_bands": "[1,2,3,4,5,6,7,8,9,10,11,12]",
                                          "batch_size": 8})


if __name__ == "__main__":
    main()
