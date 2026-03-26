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
