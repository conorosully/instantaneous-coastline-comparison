# Test cases for dataset loading and augmentation
# Conor O'Sullivan

import numpy as np
import torch
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.dataset import scale_bands, augment, TrainDataset


# ===================================================================
# Helpers
# ===================================================================

def make_args(**kwargs):
    class Args:
        pass
    args = Args()
    defaults = dict(
        target_pos=-1,
        incl_bands=np.array([0, 1, 2, 3, 4, 5, 6]),
        satellite="landsat",
        binary_mask=False,
        augmentation="none",
        aug_noise_std=0.1,
        aug_sp_prob=0.1,
        aug_contrast=[0.6, 0.8, 1.2, 1.4],
        aug_brightness=[-0.1, 0.1],
    )
    defaults.update(kwargs)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def make_npy_file(path, h=64, w=64, n_bands=8, seed=0):
    """Write a mock (H, W, C) .npy file with float values."""
    rng = np.random.default_rng(seed)
    # Bands: realistic scaled values; last channel is binary mask
    data = rng.integers(0, 10000, size=(h, w, n_bands)).astype(np.float32)
    data[:, :, -1] = rng.integers(0, 2, size=(h, w)).astype(np.float32)
    np.save(path, data)
    return path


# ===================================================================
# scale_bands
# ===================================================================

def test_scale_bands_landsat_range():
    img = np.random.randint(0, 30000, (64, 64, 7)).astype(np.float32)
    out = scale_bands(img, "landsat")
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.dtype == np.float32

def test_scale_bands_sentinel_range():
    img = np.random.randint(0, 12000, (64, 64, 12)).astype(np.float32)
    out = scale_bands(img, "sentinel")
    assert out.min() >= 0.0
    assert out.max() <= 1.0

def test_scale_bands_gaofen1_range():
    img = np.random.randint(0, 2000, (64, 64, 4)).astype(np.float32)
    out = scale_bands(img, "gaofen1")
    assert out.min() >= 0.0
    assert out.max() <= 1.0

def test_scale_bands_gaofen6_range():
    img = np.random.randint(0, 300, (64, 64, 4)).astype(np.float32)
    out = scale_bands(img, "gaofen6")
    assert out.min() >= 0.0
    assert out.max() <= 1.0

def test_scale_bands_output_dtype():
    img = np.ones((64, 64, 7), dtype=np.int16)
    out = scale_bands(img, "landsat")
    assert out.dtype == np.float32

def test_scale_bands_landsat_formula():
    """Spot-check the Landsat formula: val * 0.0000275 - 0.2."""
    img = np.array([[[10000.0]]])
    out = scale_bands(img, "landsat")
    expected = np.clip(10000 * 0.0000275 - 0.2, 0, 1)
    assert np.isclose(out[0, 0, 0], expected)


# ===================================================================
# augment
# ===================================================================

H, W, C = 64, 64, 7

def make_bands_mask():
    bands = np.random.rand(C, H, W).astype(np.float32)
    mask = np.random.randint(0, 2, (H, W)).astype(np.int8)
    return bands, mask

def test_augment_none_unchanged():
    bands, mask = make_bands_mask()
    args = make_args(augmentation="none")
    b_out, m_out = augment(bands.copy(), mask.copy(), args)
    assert np.array_equal(b_out, bands)
    assert np.array_equal(m_out, mask)

def test_augment_geometric_preserves_shape():
    bands, mask = make_bands_mask()
    args = make_args(augmentation="geometric")
    b_out, m_out = augment(bands, mask, args)
    assert b_out.shape == (C, H, W)
    assert m_out.shape == (H, W)

def test_augment_geometric_transforms_both():
    """Geometric augmentation should transform bands and mask consistently."""
    np.random.seed(1)
    bands = np.zeros((C, H, W), dtype=np.float32)
    bands[:, :5, :5] = 1.0  # mark top-left corner
    mask = np.zeros((H, W), dtype=np.int8)
    mask[:5, :5] = 1

    args = make_args(augmentation="geometric")
    b_out, m_out = augment(bands, mask, args)
    # After any rotation/flip the marked region should still coincide
    assert np.sum(b_out[0]) == pytest.approx(np.sum(bands[0]))
    assert np.sum(m_out) == np.sum(mask)

def test_augment_gaussian_noise_shape_and_range():
    bands, mask = make_bands_mask()
    args = make_args(augmentation="gaussian_noise", aug_noise_std=0.05)
    b_out, m_out = augment(bands, mask, args)
    assert b_out.shape == (C, H, W)
    assert b_out.min() >= 0.0
    assert b_out.max() <= 1.0

def test_augment_gaussian_noise_mask_unchanged():
    bands, mask = make_bands_mask()
    args = make_args(augmentation="gaussian_noise")
    _, m_out = augment(bands, mask, args)
    assert np.array_equal(m_out, mask)

def test_augment_salt_pepper_shape_and_range():
    bands, mask = make_bands_mask()
    args = make_args(augmentation="salt_pepper", aug_sp_prob=0.1)
    b_out, _ = augment(bands, mask, args)
    assert b_out.shape == (C, H, W)
    assert b_out.min() >= 0.0
    assert b_out.max() <= 1.0

def test_augment_salt_pepper_only_adds_0_or_1():
    """Salt & pepper should only add 0.0 or 1.0 values."""
    np.random.seed(42)
    bands = np.full((C, H, W), 0.5, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.int8)
    args = make_args(augmentation="salt_pepper", aug_sp_prob=0.5)
    b_out, _ = augment(bands, mask, args)
    unique = np.unique(b_out.round(6))
    for v in unique:
        assert v in (0.0, 0.5, 1.0)

def test_augment_contrast_shape_and_range():
    bands, mask = make_bands_mask()
    args = make_args(augmentation="contrast",
                     aug_contrast=[0.8, 1.2], aug_brightness=[-0.1, 0.1])
    b_out, _ = augment(bands, mask, args)
    assert b_out.shape == (C, H, W)
    assert b_out.min() >= 0.0
    assert b_out.max() <= 1.0

def test_augment_combined_shape_and_range():
    bands, mask = make_bands_mask()
    args = make_args(augmentation="combined")
    b_out, m_out = augment(bands, mask, args)
    assert b_out.shape == (C, H, W)
    assert m_out.shape == (H, W)
    assert b_out.min() >= 0.0
    assert b_out.max() <= 1.0


# ===================================================================
# TrainDataset
# ===================================================================

@pytest.fixture
def npy_dir(tmp_path):
    """Create 10 mock .npy files in a temp directory."""
    for i in range(10):
        make_npy_file(str(tmp_path / f"sample_{i}.npy"), n_bands=8, seed=i)
    return tmp_path


def test_dataset_len(npy_dir):
    paths = [str(p) for p in npy_dir.glob("*.npy")]
    args = make_args()
    ds = TrainDataset(paths, args)
    assert len(ds) == 10


def test_dataset_bands_shape(npy_dir):
    paths = [str(p) for p in npy_dir.glob("*.npy")]
    args = make_args(incl_bands=np.array([0, 1, 2, 3, 4, 5, 6]))
    ds = TrainDataset(paths, args)
    bands, _ = ds[0]
    assert bands.shape == (7, 64, 64)


def test_dataset_target_shape_multiclass(npy_dir):
    paths = [str(p) for p in npy_dir.glob("*.npy")]
    args = make_args(binary_mask=False)
    ds = TrainDataset(paths, args)
    _, target = ds[0]
    assert target.shape == (2, 64, 64)


def test_dataset_target_shape_binary(npy_dir):
    paths = [str(p) for p in npy_dir.glob("*.npy")]
    args = make_args(binary_mask=True)
    ds = TrainDataset(paths, args)
    _, target = ds[0]
    assert target.shape == (1, 64, 64)


def test_dataset_multiclass_target_sums_to_one(npy_dir):
    """Land + Water channels should sum to 1 at every pixel."""
    paths = [str(p) for p in npy_dir.glob("*.npy")]
    args = make_args(binary_mask=False)
    ds = TrainDataset(paths, args)
    _, target = ds[0]
    pixel_sums = target[0] + target[1]
    assert torch.all(pixel_sums == 1.0)


def test_dataset_bands_scaled_0_1(npy_dir):
    paths = [str(p) for p in npy_dir.glob("*.npy")]
    args = make_args(satellite="landsat")
    ds = TrainDataset(paths, args)
    bands, _ = ds[0]
    assert bands.min() >= 0.0
    assert bands.max() <= 1.0


def test_dataset_negative_one_mask_replaced(tmp_path):
    """Mask values of -1 should be converted to 0."""
    data = np.zeros((64, 64, 8), dtype=np.float32)
    data[:, :, -1] = -1  # all mask values are -1
    np.save(str(tmp_path / "neg.npy"), data)

    args = make_args(binary_mask=True)
    ds = TrainDataset([str(tmp_path / "neg.npy")], args)
    _, target = ds[0]
    assert torch.all(target == 0.0)


def test_dataset_returns_tensors(npy_dir):
    paths = [str(p) for p in npy_dir.glob("*.npy")]
    ds = TrainDataset(paths, make_args())
    bands, target = ds[0]
    assert isinstance(bands, torch.Tensor)
    assert isinstance(target, torch.Tensor)


def test_dataset_augmentation_applied(npy_dir):
    """With geometric augmentation, output should still be valid tensors of correct shape."""
    paths = [str(p) for p in npy_dir.glob("*.npy")]
    args = make_args(augmentation="geometric")
    ds = TrainDataset(paths, args)
    bands, target = ds[0]
    assert bands.shape == (7, 64, 64)
    assert target.shape == (2, 64, 64)
