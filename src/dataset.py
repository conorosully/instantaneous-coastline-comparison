import numpy as np
import torch


def scale_bands(img, satellite="landsat"):
    """Scale bands to 0-1."""
    img = img.astype("float32")
    if satellite == "landsat":
        img = np.clip(img * 0.0000275 - 0.2, 0, 1)
    elif satellite == "sentinel":
        img = np.clip(img / 10000, 0, 1)
    elif satellite == "gaofen1":
        img = np.clip(img / 1000, 0, 1)
    elif satellite == "gaofen6":
        img = np.clip(img / 255, 0, 1)
    return img


# ---------------------------------------------------------
# Runtime augmentation
# ---------------------------------------------------------

def augment(bands, mask, args):
    """
    Apply runtime augmentation to a (C, H, W) bands tensor and (H, W) mask array.
    Returns augmented bands and mask as numpy arrays.
    aug_type: "geometric" | "gaussian_noise" | "salt_pepper" | "contrast" | "combined"
    """
    aug = args.augmentation
    if aug == "none":
        return bands, mask

    if aug in ("geometric", "combined"):
        # Random rotation: 0, 90, 180, 270
        k = np.random.randint(0, 4)
        bands = np.rot90(bands, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k).copy()
        # Random horizontal flip
        if np.random.rand() > 0.5:
            bands = np.flip(bands, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        # Random vertical flip
        if np.random.rand() > 0.5:
            bands = np.flip(bands, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

    if aug in ("gaussian_noise", "combined"):
        std = args.aug_noise_std
        noise = np.random.normal(0, std, bands.shape).astype("float32")
        bands = np.clip(bands + noise, 0, 1)

    if aug in ("salt_pepper", "combined"):
        prob = args.aug_sp_prob
        rand = np.random.rand(*bands.shape)
        bands = np.where(rand < prob / 2, 0.0, bands)
        bands = np.where(rand > 1 - prob / 2, 1.0, bands)
        bands = bands.astype("float32")

    if aug in ("contrast", "combined"):
        factors = args.aug_contrast
        offsets = args.aug_brightness
        factor = np.random.choice(factors)
        offset = np.random.uniform(offsets[0], offsets[-1])
        bands = np.clip(bands * factor + offset, 0, 1).astype("float32")

    return bands, mask


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths, args):
        self.paths = paths
        self.target = args.target_pos
        self.incl_bands = args.incl_bands
        self.satellite = args.satellite
        self.binary_mask = args.binary_mask
        self.args = args

    def __getitem__(self, idx):
        instance = np.load(self.paths[idx])

        # Spectral bands: shape (H, W, C) → scale → (C, H, W)
        bands = instance[:, :, self.incl_bands].astype(np.float32)
        bands = scale_bands(bands, self.satellite)
        bands = bands.transpose(2, 0, 1)  # (C, H, W)

        # Target mask
        mask = instance[:, :, self.target].astype(np.int8)
        mask[mask == -1] = 0

        # Augmentation (operates on numpy arrays)
        bands, mask = augment(bands, mask, self.args)

        bands = torch.tensor(bands)

        if self.binary_mask:
            target = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        else:
            mask_0 = 1 - mask
            target = torch.tensor(np.array([mask_0, mask]), dtype=torch.float32)

        return bands, target

    def __len__(self):
        return len(self.paths)
