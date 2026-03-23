# Test cases for training utilities
# Conor O'Sullivan

import numpy as np
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.train import _validate_args, load_data


# ===================================================================
# Helpers
# ===================================================================

def make_args(tmp_path=None, **kwargs):
    class Args:
        pass
    args = Args()
    defaults = dict(
        model_name="test_model",
        model_type="unet",
        encoder="scratch",
        pretrained="none",
        freeze_encoder=False,
        weight_init="normal",
        satellite="landsat",
        incl_bands=np.array([0, 1, 2, 3, 4, 5, 6]),
        target_pos=-1,
        binary_mask=False,
        optimizer="adam",
        lr=0.001,
        batch_size=2,
        epochs=1,
        split=0.8,
        early_stopping=-1,
        seed=42,
        augmentation="none",
        aug_noise_std=0.1,
        aug_sp_prob=0.1,
        aug_contrast=[0.6, 0.8, 1.2, 1.4],
        aug_brightness=[-0.1, 0.1],
        train_path=str(tmp_path) if tmp_path else "../data/training/",
        valid_path=None,
        save_path=str(tmp_path / "models") if tmp_path else "../models/",
        finetune_from=None,
        sample=False,
        note="",
    )
    defaults.update(kwargs)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def make_npy_files(directory, n=10, n_bands=8):
    os.makedirs(directory, exist_ok=True)
    for i in range(n):
        data = np.random.randint(0, 10000, (64, 64, n_bands)).astype(np.float32)
        data[:, :, -1] = np.random.randint(0, 2, (64, 64)).astype(np.float32)
        np.save(os.path.join(directory, f"sample_{i}.npy"), data)


# ===================================================================
# _validate_args — valid cases
# ===================================================================

def test_validate_args_valid(tmp_path):
    args = make_args(tmp_path)
    _validate_args(args)  # should not raise

def test_validate_args_creates_save_path(tmp_path):
    save = str(tmp_path / "new_dir")
    args = make_args(tmp_path, save_path=save)
    _validate_args(args)
    assert os.path.isdir(save)

def test_validate_args_resnet_unet_valid(tmp_path):
    args = make_args(tmp_path, encoder="resnet50", model_type="unet", pretrained="imagenet")
    _validate_args(args)

def test_validate_args_all_augmentations_valid(tmp_path):
    for aug in ["none", "geometric", "gaussian_noise", "salt_pepper", "contrast", "combined"]:
        args = make_args(tmp_path, augmentation=aug)
        _validate_args(args)


# ===================================================================
# _validate_args — invalid cases
# ===================================================================

def test_validate_invalid_model_type(tmp_path):
    args = make_args(tmp_path, model_type="fancy_unet")
    with pytest.raises(ValueError):
        _validate_args(args)

def test_validate_invalid_encoder(tmp_path):
    args = make_args(tmp_path, encoder="vgg16")
    with pytest.raises(ValueError):
        _validate_args(args)

def test_validate_invalid_pretrained(tmp_path):
    args = make_args(tmp_path, pretrained="places365")
    with pytest.raises(ValueError):
        _validate_args(args)

def test_validate_bigearthnet_with_scratch_raises(tmp_path):
    args = make_args(tmp_path, encoder="scratch", pretrained="bigearthnet")
    with pytest.raises(ValueError):
        _validate_args(args)

@pytest.mark.parametrize("model_type", ["r2_unet", "r2att_unet"])
def test_validate_recurrent_with_resnet_raises(tmp_path, model_type):
    args = make_args(tmp_path, encoder="resnet18", model_type=model_type)
    with pytest.raises(ValueError):
        _validate_args(args)

@pytest.mark.parametrize("split", [0.0, 1.0, 1.5, -0.1])
def test_validate_invalid_split(tmp_path, split):
    args = make_args(tmp_path, split=split)
    with pytest.raises(ValueError):
        _validate_args(args)

def test_validate_nonexistent_valid_path(tmp_path):
    args = make_args(tmp_path, valid_path="/nonexistent/path/")
    with pytest.raises(ValueError):
        _validate_args(args)

def test_validate_invalid_augmentation(tmp_path):
    args = make_args(tmp_path, augmentation="cutmix")
    with pytest.raises(ValueError):
        _validate_args(args)


# ===================================================================
# load_data
# ===================================================================

def test_load_data_split_ratio(tmp_path):
    make_npy_files(str(tmp_path), n=10)
    args = make_args(tmp_path, split=0.8)
    train_loader, valid_loader = load_data(args)
    assert len(train_loader.dataset) == 8
    assert len(valid_loader.dataset) == 2

def test_load_data_explicit_valid_path(tmp_path):
    train_dir = str(tmp_path / "train")
    valid_dir = str(tmp_path / "valid")
    make_npy_files(train_dir, n=8)
    make_npy_files(valid_dir, n=4)
    args = make_args(tmp_path, train_path=train_dir, valid_path=valid_dir, split=0.9)
    train_loader, valid_loader = load_data(args)
    assert len(train_loader.dataset) == 8
    assert len(valid_loader.dataset) == 4

def test_load_data_sample_flag(tmp_path):
    make_npy_files(str(tmp_path), n=200)
    args = make_args(tmp_path, sample=True, split=0.8)
    train_loader, valid_loader = load_data(args)
    assert len(train_loader.dataset) + len(valid_loader.dataset) == 100

def test_load_data_batch_shape(tmp_path):
    make_npy_files(str(tmp_path), n=10)
    args = make_args(tmp_path, batch_size=4, split=0.8)
    train_loader, _ = load_data(args)
    bands, target = next(iter(train_loader))
    assert bands.shape[0] <= 4
    assert bands.shape[1] == 7  # incl_bands length
    assert target.shape[1] == 2  # multiclass

def test_load_data_seed_reproducibility(tmp_path):
    """Same seed should produce same train/val split."""
    make_npy_files(str(tmp_path), n=20)
    args1 = make_args(tmp_path, seed=42)
    args2 = make_args(tmp_path, seed=42)
    t1, _ = load_data(args1)
    t2, _ = load_data(args2)
    paths1 = t1.dataset.paths
    paths2 = t2.dataset.paths
    assert paths1 == paths2
