
import os
import glob
import numpy as np
import spyndex
import xarray as xr
import PIL
from PIL import Image, ImageEnhance

from skimage.filters import threshold_otsu
from dataset import scale_bands


band_dic = {
    "sentinel": {"coastal":0,"blue":1,"green":2,"red":3,"rededge1":4,"rededge2":5,"rededge3":6,"nir":7,"narrownir":8,"watervapour":9,"swir1":10,"swir2":11},
    "landsat":  {"blue":0,"green":1,"red":2,"nir":3,"swir1":4,"swir2":5,"thermal":6},
    "gaofen1":  {"blue":0,"green":1,"red":2,"nir":3},
    "gaofen6":  {"blue":0,"green":1,"red":2,"nir":3,"rededge1":4,"rededge2":5,"violet":6,"yellow":7},
}

# Maps band name → spyndex standard code
spyndex_dic = {
    "coastal": "A", "blue": "B", "green": "G", "red": "R",
    "rededge1": "RE1", "rededge2": "RE2", "rededge3": "RE3",
    "nir": "N", "narrownir": "NN", "watervapour": "WV",
    "swir1": "S1", "swir2": "S2", "thermal": "T",
    "violet": "V", "yellow": "Y",
}

# Maps band name → display name for plot labels / xarray coords
display_dic = {
    "coastal": "Coastal", "blue": "Blue", "green": "Green", "red": "Red",
    "rededge1": "RedEdge1", "rededge2": "RedEdge2", "rededge3": "RedEdge3",
    "nir": "NIR", "narrownir": "NarrowNIR", "watervapour": "WaterVapour",
    "swir1": "SWIR1", "swir2": "SWIR2", "thermal": "Thermal",
    "violet": "Violet", "yellow": "Yellow",
}


def edge_from_mask(mask):
    """Get edge map from mask"""

    dy, dx = np.gradient(mask)
    grad = np.abs(dx) + np.abs(dy)
    edge = np.array([grad > 0])[0]
    edge = edge.astype(np.uint8)

    return edge

def get_threshold(index, threshold):
    """Get NDWI threshold prediction"""
    img = np.nan_to_num(index.copy(), nan=0.0)

    if threshold == 'otsu':
        threshold = threshold_otsu(img)

    img[img >= threshold] = 1
    img[img < threshold] = 0
    return img


def get_index(bands, index="MNDWI",satellite='sentinel'):

    img = bands.copy()

    """Add indices to image"""

    if satellite == 'landsat':

        channels = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "Thermal"]
        standards = ["B", "G", "R", "N", "S1", "S2", "T"]

        img = img[:, :, [0, 1, 2, 3, 4, 5, 6]]

    elif satellite == 'sentinel':

        channels = ["Coastal", "Blue", "Green", "Red", "RedEdge1", "RedEdge2", "RedEdge3", "NIR", "NarrowNIR", "WaterVapour", "SWIR1", "SWIR2"]
        standards = ["A","B", "G", "R", "RE1", "RE2", "RE3", "N", "NN", "WV", "S1", "S2"]

        img = img[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

    elif satellite == 'gaofen1':

        channels = ["Blue", "Green", "Red", "NIR"]
        standards = ["B", "G", "R", "N"]

        img = img[:, :, [0, 1, 2, 3]]

    elif satellite == 'gaofen6':
        channels = [ "Blue", "Green", "Red", "NIR","RedEdge1", "RedEdge2", "Violet","Yellow"]
        standards = [ "B", "G", "R", "N", "RE1", "RE2", "V", "Y"]


        img = img[:, :, [0,1,2,3,4,5,6,7]]

        
    img = img.astype(np.float32)
    img = scale_bands(img,satellite=satellite)

    da = xr.DataArray(img, dims=("x", "y", "band"), coords={"band": channels})

    params = {
        standard: da.sel(band=channel) for standard, channel in zip(standards, channels)
    }

    idx = spyndex.computeIndex(index=[index], params=params)

    idx = np.array(idx)

    return idx


def predict_index(bands, satellite, index="MNDWI", threshold="otsu"):
    """
    Predict a binary water mask from raw spectral bands using an index + threshold.

    bands     : (H, W, C) numpy array of raw (unscaled) band values
    satellite : "landsat" | "sentinel" | "gaofen1" | "gaofen6"
    index     : any spyndex-supported index name (default "MNDWI")
    threshold : "otsu" for Otsu's method, or a float for a fixed cutoff

    Returns: (H, W) uint8 array — 1 = water, 0 = land
    """
    idx  = get_index(bands, index=index, satellite=satellite)
    mask = get_threshold(idx, threshold=threshold)
    return mask.astype(np.uint8)


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

DATASET_CONFIG = {
    "LICS":             {"incl_bands": [0,1,2,3,4,5,6],             "target_pos": -2, "satellite": "landsat"},
    "SWED":             {"incl_bands": [0,1,2,3,4,5,6,7,8,9,10,11], "target_pos": -1, "satellite": "sentinel"},
    "SANet_processed":  {"incl_bands": [0,1,2,3],                    "target_pos": -1, "satellite": "gaofen1"},
    "TCUNet_processed": {"incl_bands": [0,1,2,3,4,5,6,7],           "target_pos": -1, "satellite": "gaofen6"},
}


def load_dataset(name, path, sample=None, overrides=None):
    """
    Load inputs and targets for a dataset split.

    name      : dataset name — one of DATASET_CONFIG keys
    path      : directory containing .npy files
    sample    : if set, randomly select this many files (reproducible seed)
    overrides : dict of config keys to override, e.g. {"target_pos": -1}

    Returns: inputs  — list of (H, W, C) raw band arrays
             targets — list of (H, W) int mask arrays
             satellite — satellite string for this dataset
    """
    cfg        = {**DATASET_CONFIG[name], **(overrides or {})}
    incl_bands = cfg["incl_bands"]
    target_pos = cfg["target_pos"]

    files = sorted(glob.glob(os.path.join(path, "*.npy")))
    if sample is not None and sample < len(files):
        rng   = np.random.default_rng(42)
        files = list(rng.choice(files, size=sample, replace=False))

    data  = [np.load(f) for f in files]

    inputs  = [d[:, :, incl_bands] for d in data]
    targets = [np.where(d[:, :, target_pos] == -1, 0, d[:, :, target_pos]).astype(int) for d in data]

    return inputs, targets, cfg["satellite"]


def load_all_datasets(paths, sample=None, overrides=None):
    """
    Load inputs and targets for multiple datasets in one call.

    paths     : dict mapping dataset name to its directory path
                e.g. {"LICS": "../../data/LICS/test",
                      "SWED": "../data/SWED/test", ...}
    sample    : if set, randomly select this many files per dataset
    overrides : dict mapping dataset name to config overrides
                e.g. {"LICS": {"target_pos": -1}}

    Returns: dict {name: {"inputs": [...], "targets": [...], "satellite": "..."}}
    """
    overrides = overrides or {}
    return {
        name: dict(zip(
            ("inputs", "targets", "satellite"),
            load_dataset(name, path, sample=sample, overrides=overrides.get(name)),
        ))
        for name, path in paths.items()
    }


# ---------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------

def training_data_check(paths, args, n_sample=50):
    """Sense-check band scaling and mask values on a sample of training files."""
    rng = np.random.default_rng(42)
    sample = rng.choice(len(paths), size=min(n_sample, len(paths)), replace=False)

    band_min = None
    band_max = None
    mask_vals = set()

    for i in sample:
        arr = np.load(paths[i])
        bands = arr[:, :, args.incl_bands].astype(np.float32)
        bands = scale_bands(bands, args.satellite)
        mask = arr[:, :, args.target_pos]
        if band_min is None:
            band_min = bands.min(axis=(0, 1))
            band_max = bands.max(axis=(0, 1))
        else:
            band_min = np.minimum(band_min, bands.min(axis=(0, 1)))
            band_max = np.maximum(band_max, bands.max(axis=(0, 1)))
        mask_vals.update(np.unique(mask).tolist())

    n_bands = len(args.incl_bands)
    sat_bands = list(band_dic.get(args.satellite, {}).keys())
    names = [sat_bands[i] for i in args.incl_bands] if sat_bands else [f"band_{i}" for i in args.incl_bands]

    print(f"  Satellite: {args.satellite} | bands ({n_bands}): {names}")
    scale_ok = all(band_min[i] >= -0.01 and band_max[i] <= 1.01 for i in range(n_bands))
    print(f"  Scaled band min/max ({'OK' if scale_ok else 'WARNING out of [0,1]'}):")
    for i in range(n_bands):
        flag = "" if -0.01 <= band_min[i] and band_max[i] <= 1.01 else "  <- out of range"
        print(f"    {names[i]}: [{band_min[i]:.4f}, {band_max[i]:.4f}]{flag}")
    mask_ok = mask_vals <= {0, 1}
    extra = mask_vals - {0, 1}
    print(f"  Target unique values: {sorted(mask_vals)} ({'OK' if mask_ok else f'WARNING unexpected: {extra}'})")


def dataset_summary(paths, satellite, split_name=""):
    """Print basic info about a set of .npy files (count, shape, per-band min/max)."""
    n = len(paths)
    if n == 0:
        print(f"  {split_name}: no files found")
        return

    sample = np.load(paths[0])
    shape = sample.shape
    n_bands = shape[-1] - 1  # last channel is mask

    # Accumulate min/max over all files
    band_min = np.full(n_bands, np.inf)
    band_max = np.full(n_bands, -np.inf)
    for p in paths:
        arr = np.load(p)
        bands = arr[..., :n_bands].astype(np.float32)
        bands = scale_bands(bands, satellite)
        band_min = np.minimum(band_min, bands.min(axis=(0, 1)))
        band_max = np.maximum(band_max, bands.max(axis=(0, 1)))

    # Collect unique mask values across all files
    mask_vals = set()
    for p in paths:
        mask_vals.update(np.unique(np.load(p)[..., -1]).tolist())

    label = f"  [{split_name}]" if split_name else " "
    print(f"{label} {n} files | shape {shape} | mask unique: {sorted(mask_vals)} | scaled band min/max:")
    for i in range(n_bands):
        print(f"    band {i}: [{band_min[i]:.4f}, {band_max[i]:.4f}]")


def show_examples(paths, n=4, satellite="gaofen1", rgb_bands=None,
                  show_bands=False, band_names=None, seed=0):
    """Display n random examples.

    show_bands=False (default): RGB composite + mask (2 columns).
    show_bands=True: individual bands in grayscale + mask (n_bands+1 columns).
    band_names: list of strings for column titles when show_bands=True.
    """
    import matplotlib.pyplot as plt

    if rgb_bands is None:
        rgb_bands = [2, 1, 0]  # Red, Green, Blue

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=min(n, len(paths)), replace=False)

    # Determine number of columns from first file
    sample = np.load(paths[indices[0]])
    n_bands = sample.shape[-1] - 1
    n_cols = n_bands + 1 if show_bands else 2

    if band_names is None:
        if satellite in band_dic:
            band_names = [display_dic[name] for name in band_dic[satellite]]
        else:
            band_names = [f"Band {i}" for i in range(n_bands)]

    fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n))
    if n == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        arr = np.load(paths[idx])
        bands = arr[..., :n_bands].astype(np.float32)
        bands = scale_bands(bands, satellite)
        mask = arr[..., -1]

        if show_bands:
            for col in range(n_bands):
                axes[row][col].imshow(bands[..., col], cmap="gray")
                axes[row][col].set_title(f"{band_names[col]}  ({idx})", fontsize=9)
                axes[row][col].axis("off")
            axes[row][n_bands].imshow(mask, cmap="Blues", vmin=0, vmax=1)
            axes[row][n_bands].set_title(f"Mask  ({idx})", fontsize=9)
            axes[row][n_bands].axis("off")
        else:
            rgb = np.clip(bands[..., rgb_bands], 0, 1)
            axes[row][0].imshow(rgb)
            axes[row][0].set_title(f"RGB  ({idx})", fontsize=9)
            axes[row][0].axis("off")
            axes[row][1].imshow(mask, cmap="Blues", vmin=0, vmax=1)
            axes[row][1].set_title(f"Mask  ({idx})", fontsize=9)
            axes[row][1].axis("off")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Data visualization
# ---------------------------------------------------------

def get_rgb(img, r=2,g=1,b=0, contrast=1,satellite='sentinel'):
    """Convert a stacked array of bands to RGB"""

    if img.shape[0] > img.shape[2]:
        #  H x W x B -> B x H x W
        img = np.transpose(img, (2,0,1))

    r = img[r]
    g = img[g]
    b = img[b]

    rgb = np.stack([r, g, b], axis=-1)
    rgb = rgb.astype(np.float32)

    rgb = scale_bands(rgb, satellite=satellite)
    rgb = np.clip(rgb, 0, contrast) / contrast

    # convert to 255
    rgb = (rgb * 255).astype(np.uint8) 

    return rgb

def enhance_rgb(rgb_array,factor=1.5):
    """Enhance the RGB image."""
    
    RGB = PIL.Image.fromarray(rgb_array)

    converter = ImageEnhance.Color(RGB)

    RGB = converter.enhance(factor)
    RGB = np.array(RGB)

    return RGB