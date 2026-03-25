"""
Download script for coastal monitoring datasets.

Usage:
    python download.py --save_path ./data --dataset all --todo download
    python download.py --save_path ./data --dataset LICS --todo download
    python download.py --save_path ./data --dataset SWED --todo download

Datasets:
    LICS   - https://zenodo.org/records/13742222
    SWED   - https://openmldata.ukho.gov.uk/
    SANet  - https://ieee-dataport.org/open-access/sea-landsegmentationdataset
    TCUNet - https://aistudio.baidu.com/datasetdetail/230558
"""

import argparse
import glob
import os
import zipfile

import requests
from tqdm import tqdm

DATASETS = {
    "LICS": {
        "url": "https://zenodo.org/records/13742222/files/training.zip?download=1",
        "filename": "LICS.zip",
        "source": "https://zenodo.org/records/13742222",
        "extra_files": [
            {"url": "https://zenodo.org/records/13742222/files/test.zip?download=1",     "filename": "LICS_test.zip"},
            {"url": "https://zenodo.org/records/13742222/files/finetune.zip?download=1", "filename": "LICS_finetune.zip"},
        ],
    },
    "SWED": {
        "url": "https://ukho-openmldata.s3.eu-west-2.amazonaws.com/SWED.zip",
        "filename": "SWED.zip",
        "source": "https://openmldata.ukho.gov.uk/",
    },
    "SANet": {
        "url": "https://drive.google.com/uc?id=1IiV8IRJ06-qYVPbi6dGqiNxo4qJHyuJP",
        "filename": "SANet.zip",
        "source": "https://ieee-dataport.org/open-access/sea-landsegmentationdataset",
        "gdrive": True,
        "unzip_nested": True,
        "convert_npy": True,
    },
    "TCUNet": {
        "url": "https://drive.google.com/uc?id=1gD5pA8Df7qoqe4eDTqgqd__sXvOsSPQt",
        "filename": "TCUNet.zip",
        "source": "https://aistudio.baidu.com/datasetdetail/230558",
        "gdrive": True,
        "convert_npy": True,
    },
}


# --- Download / unzip ---


def download_file(url, dest_path, gdrive=False):
    """Download a file from a URL to dest_path with a progress bar."""
    if gdrive:
        try:
            import gdown
        except ImportError:
            raise ImportError(
                "gdown is required for Google Drive downloads. "
                "Install it with: pip install gdown"
            )
        gdown.download(url, dest_path, quiet=False)
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def unzip_file(zip_path, extract_to):
    """Unzip a file into extract_to directory."""
    print(f"  Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print(f"  Extracted to: {extract_to}")


def unzip_nested(dataset_dir):
    """Unzip any zip files found in subdirectories of dataset_dir."""
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if fname.endswith(".zip"):
                zip_path = os.path.join(root, fname)
                print(f"  Extracting nested zip: {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(root)
                os.remove(zip_path)
                print(f"  Extracted to: {root}")


# --- SANet .npy conversion ---


def _read_image(path):
    """Read a multi-band GeoTIFF and return (H, W, C) uint16 array."""
    import rasterio
    with rasterio.open(path) as src:
        array = src.read()
    return array.transpose(1, 2, 0)


def _read_mask(path):
    """Read a mask GeoTIFF and return (H, W, 1) uint8 array. 1=water, 0=land."""
    import numpy as np
    import rasterio
    with rasterio.open(path) as src:
        array = src.read()
    array = np.where(array == 255, 1, 0)
    array = 1 - array
    return array.astype(np.uint8).transpose(1, 2, 0)


def _get_train_valid_paths(base_dir, split):
    image_paths = sorted(glob.glob(os.path.join(base_dir, split, "*.tif")))
    mask_paths = [
        os.path.join(base_dir, f"{split}t", os.path.basename(p))
        for p in image_paths
    ]
    missing = [p for p in mask_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing masks: {missing[:5]}")
    return image_paths, mask_paths


def _get_test_paths(base_dir):
    image_paths, mask_paths = [], []
    for i in range(1, 13):
        for img_path in sorted(glob.glob(os.path.join(base_dir, f"test{i}", f"test{i}_*.tif"))):
            fname = os.path.basename(img_path)
            mask_path = os.path.join(base_dir, f"test{i}t", fname.replace(f"test{i}_", f"test{i}t_"))
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
    return image_paths, mask_paths


def _convert_split(image_paths, mask_paths, out_dir):
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        image = _read_image(img_path)
        mask = _read_mask(mask_path)
        combined = np.concatenate([image, mask], axis=-1)
        np.save(os.path.join(out_dir, f"{stem}.npy"), combined)


def convert_sanet(dataset_dir, out_root):
    """Convert SANet .tif files to per-instance .npy files (4 bands + mask)."""
    src_dir = os.path.join(dataset_dir, "croped_images")
    for split in ("train", "valid"):
        print(f"\n  [{split}]")
        image_paths, mask_paths = _get_train_valid_paths(src_dir, split)
        _convert_split(image_paths, mask_paths, os.path.join(out_root, split))
    print("\n  [test]")
    image_paths, mask_paths = _get_test_paths(src_dir)
    _convert_split(image_paths, mask_paths, os.path.join(out_root, "test"))


def _read_mask_threshold(path):
    """Read a mask GeoTIFF scaled 0-255 and return (H, W, 1) uint8. 1=water, 0=land."""
    import numpy as np
    import rasterio
    with rasterio.open(path) as src:
        array = src.read()
    array = (array >= 128).astype(np.uint8)
    return array.transpose(1, 2, 0)


def convert_tcunet(dataset_dir, out_root):
    """Convert TCUNet .tif files to per-instance .npy files (8 bands + mask)."""
    for split, img_subdir, mask_subdir in [
        ("train", "train/images", "train/labels"),
        ("test",  "test/images",  "test/mndwi"),
    ]:
        print(f"\n  [{split}]")
        image_paths = sorted(glob.glob(os.path.join(dataset_dir, img_subdir, "*.tif")))
        mask_paths  = [
            os.path.join(dataset_dir, mask_subdir, os.path.basename(p))
            for p in image_paths
        ]
        import numpy as np
        os.makedirs(os.path.join(out_root, split), exist_ok=True)
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            image = _read_image(img_path)
            mask  = _read_mask_threshold(mask_path)
            combined = np.concatenate([image, mask], axis=-1)
            np.save(os.path.join(out_root, split, f"{stem}.npy"), combined)


# --- Summary ---


def summarise_dataset(name, dataset_dir):
    """Print file counts for each immediate subfolder of dataset_dir."""
    print(f"\n[{name}] Summary:")
    if not os.path.exists(dataset_dir):
        print("  Directory not found.")
        return
    subdirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    if subdirs:
        for subdir in subdirs:
            path = os.path.join(dataset_dir, subdir)
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"  {subdir}/: {count} files")
    else:
        count = len([f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))])
        print(f"  {count} files")


# --- Pipeline ---


def process_dataset(name, config, save_path, todo):
    dataset_dir = os.path.join(save_path, name)
    zip_path = os.path.join(save_path, config["filename"])

    os.makedirs(dataset_dir, exist_ok=True)

    if todo in ("download", "both"):
        print(f"\n[{name}] Downloading...")
        print(f"  Source: {config['source']}")
        try:
            download_file(config["url"], zip_path, gdrive=config.get("gdrive", False))
            print(f"  Downloaded to: {zip_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print(f"  You can download {name} manually from: {config['source']}")
            return
        for extra in config.get("extra_files", []):
            extra_zip = os.path.join(save_path, extra["filename"])
            try:
                download_file(extra["url"], extra_zip)
                print(f"  Downloaded to: {extra_zip}")
            except Exception as e:
                print(f"  Download failed for {extra['filename']}: {e}")

    if todo in ("process", "both"):
        if not os.path.exists(zip_path):
            print(f"\n[{name}] No zip file found at {zip_path}, skipping.")
            return
        print(f"\n[{name}] Processing...")
        unzip_file(zip_path, dataset_dir)
        for extra in config.get("extra_files", []):
            extra_zip = os.path.join(save_path, extra["filename"])
            if os.path.exists(extra_zip):
                unzip_file(extra_zip, dataset_dir)
            else:
                print(f"  No zip found for {extra['filename']}, skipping.")
        if config.get("unzip_nested"):
            print(f"\n[{name}] Extracting nested zips...")
            unzip_nested(dataset_dir)
        if config.get("convert_npy"):
            out_root = os.path.join(save_path, f"{name}_processed")
            print(f"\n[{name}] Converting to .npy -> {out_root}")
            if name == "SANet":
                convert_sanet(dataset_dir, out_root)
            elif name == "TCUNet":
                convert_tcunet(dataset_dir, out_root)
            summarise_dataset(f"{name}_processed", out_root)
        else:
            summarise_dataset(name, dataset_dir)


def main():
    parser = argparse.ArgumentParser(description="Download coastal monitoring datasets.")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory to save downloaded datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["LICS", "SWED", "SANet", "TCUNet", "all"],
        help="Dataset to download (default: all).",
    )
    parser.add_argument(
        "--todo",
        type=str,
        default="download",
        choices=["download", "process", "both"],
        help=(
            "Action to perform: 'download' fetches the zip files, "
            "'process' unzips downloaded files, "
            "'both' does download then process."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"Save path : {args.save_path}")
    print(f"Datasets  : {', '.join(targets)}")
    print(f"Action    : {args.todo}")

    for name in targets:
        process_dataset(name, DATASETS[name], args.save_path, args.todo)

    print("\nDone.")
    print("\nIf any downloads failed, visit the original sources:")
    for name, config in DATASETS.items():
        if name in targets:
            print(f"  {name}: {config['source']}")


if __name__ == "__main__":
    main()
