# instantaneous-coastline-comparison
A comparison of simple spectral indices and advanced deep learning models applied to detect the instantaneous coastline.

## Datasets

| Dataset | Satellite | Bands | Splits | Source |
|---------|-----------|-------|--------|--------|
| LICS | Sentinel-2 | 12 | train, test, finetune | [Zenodo](https://zenodo.org/records/13742222) |
| SWED | Sentinel-2 | 12 | train, test | [UKHO](https://openmldata.ukho.gov.uk/) |
| SANet | Gaofen-1 | 4 | train, valid, test | [IEEE DataPort](https://ieee-dataport.org/open-access/sea-landsegmentationdataset) |
| TCUNet | Gaofen-6 | 8 | train, test | [Baidu AI Studio](https://aistudio.baidu.com/datasetdetail/230558) |

> **Note:** SANet and TCUNet are hosted on Google Drive mirrors as the original sources require login.

Each dataset is processed into `.npy` files of shape `(H, W, C+1)` where the last channel is the binary water/land mask.

## Setup

Install the required dependencies:

```bash
pip install requests tqdm gdown rasterio
```

## Downloading & Processing

The `src/download.py` script handles downloading and processing. It accepts three arguments:

| Flag | Options | Description |
|------|---------|-------------|
| `--save_path` | any path | Directory to save datasets |
| `--dataset` | `LICS`, `SWED`, `SANet`, `TCUNet`, `all` | Dataset to download (default: `all`) |
| `--todo` | `download`, `process`, `both` | Action to perform |

- **`download`** — fetches the zip files
- **`process`** — unzips and converts to `.npy` format
- **`both`** — download then process in one step

### Example commands

Download and process all datasets in one step:
```bash
python src/download.py --save_path ./data --dataset all --todo both
```

Download a single dataset:
```bash
python src/download.py --save_path ./data --dataset LICS --todo download
```

Process (unzip/convert) a previously downloaded dataset:
```bash
python src/download.py --save_path ./data --dataset SWED --todo process
```

Download and process each dataset separately:
```bash
python src/download.py --save_path ./data --dataset LICS --todo both
python src/download.py --save_path ./data --dataset SWED --todo both
python src/download.py --save_path ./data --dataset SANet --todo both
python src/download.py --save_path ./data --dataset TCUNet --todo both
```

## Training a Model

`src/train.py` trains a single segmentation model from CLI arguments. It picks the model architecture, loads a training/validation split, runs the training loop with early stopping, and (optionally) sweeps over several learning rates in one call, saving only the best-performing weights.

Required arguments:

| Flag | Description |
|------|-------------|
| `--model_name` | Name used for the saved `.pth`/`.json` output files |
| `--satellite` | One of `landsat`, `sentinel`, `gaofen1`, `gaofen6` |

Commonly used optional arguments:

| Flag | Options | Description |
|------|---------|-------------|
| `--model_type` | `unet`, `r2_unet`, `att_unet`, `r2att_unet`, `swed_unet` | Model architecture (default: `unet`) |
| `--encoder` | `scratch`, `resnet18`, `resnet50`, `resnet101` | Encoder backbone (default: `scratch`) |
| `--pretrained` | `none`, `imagenet`, `bigearthnet` | Pretrained encoder weights (requires a ResNet `--encoder`) |
| `--freeze_encoder` | flag | Freeze the encoder weights during training |
| `--incl_bands` | e.g. `"[1,2,3,4,5,6,7]"` | 1-indexed input band positions |
| `--binary_mask` | flag | Use a single-channel BCE mask instead of the default 2-channel cross-entropy target |
| `--optimizer` | `adam`, `adamw`, `sgd` | Optimiser (default: `adam`) |
| `--lr` | one or more floats | Learning rate(s). Pass several to run a sweep, e.g. `--lr 0.01 0.001 0.0001` — the best is kept automatically |
| `--batch_size`, `--epochs`, `--split` | | Standard training loop settings |
| `--early_stopping` | int | Patience in epochs (`-1` disables early stopping) |
| `--augmentation` | `none`, `geometric`, `gaussian_noise`, `salt_pepper`, `contrast`, `combined` | Augmentation strategy |
| `--train_path`, `--valid_path` | path | Training data directory, and an optional explicit validation directory (otherwise `--split` is used to carve one out of `--train_path`) |
| `--save_path` | path | Directory to write the trained `.pth` weights and `.json` config/metrics |
| `--finetune_from` | path | Local `.pth` state dict to initialise the model from before training |
| `--sample` | flag | Use only the first 100 training files, for a quick smoke test |
| `--device` | e.g. `cuda`, `mps`, `cpu` | Training device (default: `cuda`) |

Run `python src/train.py --help` for the full list of arguments (loss weighting, weight initialisation, augmentation parameters, etc.).

Each run writes two files to `--save_path`: `<model_name>.pth` (best model weights) and `<model_name>.json` (the full config plus per-epoch validation losses for every learning rate tried).

### Example commands

Train a basic U-Net on LICS:
```bash
python src/train.py --model_name lics_unet --satellite landsat \
    --train_path ./data/LICS/train --save_path ./models
```

Sweep learning rates and use SGD:
```bash
python src/train.py --model_name lics_unet_sgd --satellite landsat \
    --train_path ./data/LICS/train --save_path ./models \
    --optimizer sgd --lr 0.1 0.01 0.001
```

Fine-tune a ResNet50 encoder pretrained on ImageNet, with the encoder frozen:
```bash
python src/train.py --model_name lics_resnet50_imagenet --satellite landsat \
    --train_path ./data/LICS/finetune --save_path ./models \
    --encoder resnet50 --pretrained imagenet --freeze_encoder
```

Continue training from a previously saved checkpoint:
```bash
python src/train.py --model_name lics_unet_ft --satellite landsat \
    --train_path ./data/LICS/finetune --save_path ./models \
    --finetune_from ./models/lics_unet.pth
```

Quick smoke test on a small sample:
```bash
python src/train.py --model_name test_run --satellite landsat \
    --train_path ./data/LICS/train --save_path ./models \
    --sample --epochs 2
```

## Running Experiments

`src/experiments.py` wraps `train.py` to run the predefined experiment suites used in this project. Each experiment trains a batch of models by calling `run_experiment()` (the Python-dict equivalent of the `train.py` CLI) in a loop, and results are written as usual to `--save_path` (one `.pth`/`.json` pair per model).

| Experiment | Description |
|------------|-------------|
| 1 | U-Net hyperparameter sweep (optimizer × lr) across LICS, SWED, SANet, TCUNet |
| 2 | Architecture comparison (`unet`, `r2_unet`, `att_unet`, `r2att_unet`, `swed_unet`) × optimizer, across datasets |
| 3 | Augmentation comparison (`none`, `geometric`, `gaussian_noise`, `salt_pepper`, `contrast`, `combined`) × optimizer, on LICS and/or SWED |
| 4 | Fine-tuning comparison: continuing from an experiment-2 checkpoint vs. ImageNet/BigEarthNet-pretrained ResNet encoders (frozen and fine-tuned) |
| 5 | U-Net trained on a reduced, dataset-specific band subset, across datasets |

Key arguments:

| Flag | Description |
|------|-------------|
| `--experiment` | `1`–`5` to run a single experiment; omit to run all |
| `--train_path` | LICS training data (needed for experiments 1, 2) |
| `--finetune_path` / `--swed_finetune_path` | LICS / SWED fine-tuning data (experiment 3) |
| `--scratch_path` | Directory containing the SWED, SANet_processed, TCUNet_processed subfolders (needed for experiments 1, 2, 4, 5) |
| `--save_path` | Directory to save trained models and configs |
| `--exp1_dataset`, `--exp2_dataset`, `--exp4_dataset`, `--exp5_dataset` | Restrict the given experiment to a single dataset |
| `--exp2_models` | Restrict experiment 2 to specific architectures, e.g. `--exp2_models unet att_unet` |
| `--exp4_models_dir` | Directory of experiment-2 `.pth` files used as fine-tuning starting points (required for experiment 4) |
| `--exp4_groups` | Restrict experiment 4 to specific groups: `unet`, `imagenet`, `bigearthnet` |

Run `python src/experiments.py --help` for the full list of arguments.

### Example commands

Run every experiment:
```bash
python src/experiments.py \
    --train_path ./data/LICS/train \
    --finetune_path ./data/LICS/finetune \
    --swed_finetune_path ./data/SWED/swed_finetune \
    --scratch_path ./data \
    --save_path ./models
```

Run only experiment 2 (architecture comparison) on a single dataset:
```bash
python src/experiments.py --experiment 2 \
    --train_path ./data/LICS/train --scratch_path ./data \
    --save_path ./models --exp2_dataset LICS
```

Run experiment 4 (fine-tuning), using experiment 2's checkpoints as the starting point:
```bash
python src/experiments.py --experiment 4 \
    --scratch_path ./data --save_path ./models \
    --exp4_models_dir ./models/exp2 --exp4_dataset LICS
```

### Evaluation mode

`experiments.py` also evaluates saved models (or a spectral-index baseline) against a test set and writes per-image results to a CSV.

Evaluate all trained models in `--models_dir`:
```bash
python src/experiments.py --evaluate \
    --models_dir ./models \
    --lics_test ./data/LICS/test --swed_test ./data/SWED/test \
    --sanet_test ./data/SANet_processed/test --tcunet_test ./data/TCUNet_processed/test \
    --output_csv ./results/results.csv
```

Evaluate a spectral index baseline (e.g. NDWI) with fixed thresholds and Otsu's method:
```bash
python src/experiments.py --evaluate_index --index NDWI \
    --lics_test ./data/LICS/test --swed_test ./data/SWED/test \
    --output_csv ./results/results_ndwi.csv
```
