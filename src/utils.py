
import numpy as np
import spyndex
import xarray as xr

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

def get_threshold(index,threshold):
    """Get NDWI threshold predcition"""
    img = index.copy()

    if threshold == 'otsu':
        threshold = threshold_otsu(img)

    """Threshold an image"""
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


# ---------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------

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