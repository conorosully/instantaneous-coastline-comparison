# Test cases for network architectures
# Conor O'Sullivan

import types
import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.network import (
    unet, r2_unet, att_unet, r2att_unet,
    init_weights, conv_block, RRCNN_block, up_conv, attention_block,
    get_model, _adapt_first_conv, ENCODER_CHANNELS,
)


# ===================================================================
# Weight initialisation
# ===================================================================

def test_init_weights():
    model = unet(input_channels=7, output_channels=2)
    init_weights(model, init_type="kaiming")
    for name, param in model.named_parameters():
        if "weight" in name:
            assert torch.any(param.data != 0)

def test_invalid_init_type():
    model = unet()
    with pytest.raises(NotImplementedError):
        init_weights(model, init_type="wrong-type")


# ===================================================================
# Scratch encoder forward passes
# ===================================================================

def test_unet_forward():
    model = unet()
    assert model(torch.randn(1, 3, 256, 256)).shape == (1, 1, 256, 256)

    model = unet(input_channels=7, output_channels=2)
    assert model(torch.randn(1, 7, 128, 128)).shape == (1, 2, 128, 128)

def test_r2_unet_forward():
    model = r2_unet()
    assert model(torch.randn(1, 3, 256, 256)).shape == (1, 1, 256, 256)

    model = r2_unet(input_channels=7, output_channels=2)
    assert model(torch.randn(1, 7, 128, 128)).shape == (1, 2, 128, 128)

def test_att_unet_forward():
    model = att_unet()
    assert model(torch.randn(1, 3, 256, 256)).shape == (1, 1, 256, 256)

    model = att_unet(input_channels=7, output_channels=2)
    assert model(torch.randn(1, 7, 128, 128)).shape == (1, 2, 128, 128)

def test_r2att_unet_forward():
    model = r2att_unet()
    assert model(torch.randn(1, 3, 256, 256)).shape == (1, 1, 256, 256)

    model = r2att_unet(input_channels=7, output_channels=2)
    assert model(torch.randn(1, 7, 128, 128)).shape == (1, 2, 128, 128)


# ===================================================================
# Scratch encoder backward passes
# ===================================================================

def test_unet_backward_pass():
    model = unet(input_channels=7, output_channels=2)
    loss = model(torch.randn(1, 7, 128, 128)).mean()
    loss.backward()
    assert len([p.grad for p in model.parameters() if p.grad is not None]) > 0

def test_r2_unet_backward_pass():
    model = r2_unet(input_channels=7, output_channels=2)
    loss = model(torch.randn(1, 7, 128, 128)).mean()
    loss.backward()
    assert len([p.grad for p in model.parameters() if p.grad is not None]) > 0

def test_att_unet_backward_pass():
    model = att_unet(input_channels=7, output_channels=2)
    loss = model(torch.randn(1, 7, 128, 128)).mean()
    loss.backward()
    assert len([p.grad for p in model.parameters() if p.grad is not None]) > 0

def test_r2att_unet_backward_pass():
    model = r2att_unet(input_channels=7, output_channels=2)
    loss = model(torch.randn(1, 7, 128, 128)).mean()
    loss.backward()
    assert len([p.grad for p in model.parameters() if p.grad is not None]) > 0


# ===================================================================
# Building blocks
# ===================================================================

def test_conv_block():
    block = conv_block(3, 64)
    assert block(torch.randn(1, 3, 128, 128)).shape == (1, 64, 128, 128)

def test_attention_block():
    att = attention_block(F_g=64, F_l=64, F_int=32)
    out = att(g=torch.randn(1, 64, 64, 64), x=torch.randn(1, 64, 64, 64))
    assert out.shape == (1, 64, 64, 64)

def test_rrcnn_block():
    block = RRCNN_block(64, 128)
    assert block(torch.randn(1, 64, 64, 64)).shape == (1, 128, 64, 64)


# ===================================================================
# get_model factory — valid combinations
# ===================================================================

def test_get_model_scratch_all_types():
    for model_type in ["unet", "r2_unet", "att_unet", "r2att_unet"]:
        model = get_model("scratch", model_type, 7, 2)
        assert model(torch.randn(1, 7, 128, 128)).shape == (1, 2, 128, 128)

@pytest.mark.parametrize("encoder", ["resnet18", "resnet50", "resnet101"])
def test_get_model_resnet_unet(encoder):
    model = get_model(encoder, "unet", 7, 2, pretrained="none")
    assert model(torch.randn(1, 7, 128, 128)).shape == (1, 2, 128, 128)

@pytest.mark.parametrize("encoder", ["resnet18", "resnet50", "resnet101"])
def test_get_model_resnet_att_unet(encoder):
    model = get_model(encoder, "att_unet", 7, 2, pretrained="none")
    assert model(torch.randn(1, 7, 128, 128)).shape == (1, 2, 128, 128)

def test_get_model_resnet_binary_output():
    model = get_model("resnet18", "unet", 7, 1, pretrained="none")
    assert model(torch.randn(1, 7, 128, 128)).shape == (1, 1, 128, 128)

def test_get_model_resnet_variable_in_channels():
    """Encoder first conv should adapt to any number of input channels."""
    for in_ch in [1, 4, 12]:
        model = get_model("resnet18", "unet", in_ch, 2, pretrained="none")
        assert model(torch.randn(1, in_ch, 128, 128)).shape == (1, 2, 128, 128)


# ===================================================================
# get_model factory — invalid combinations
# ===================================================================

@pytest.mark.parametrize("model_type", ["r2_unet", "r2att_unet"])
def test_get_model_recurrent_with_resnet_raises(model_type):
    with pytest.raises(ValueError, match="recurrent"):
        get_model("resnet18", model_type, 7, 2, pretrained="none")

def test_get_model_bigearthnet_with_scratch_raises():
    with pytest.raises(ValueError):
        get_model("scratch", "unet", 7, 2, pretrained="bigearthnet")


# ===================================================================
# freeze_encoder
# ===================================================================

def test_freeze_encoder_freezes_deep_layers():
    model = get_model("resnet18", "unet", 7, 2, pretrained="none", freeze_encoder=True)
    # Layers beyond conv1 should be frozen
    for param in model.encoder.layer4.parameters():
        assert not param.requires_grad

def test_freeze_encoder_keeps_conv1_trainable():
    model = get_model("resnet18", "unet", 7, 2, pretrained="none", freeze_encoder=True)
    # conv1 is stage0[0]
    for param in model.encoder.stage0[0].parameters():
        assert param.requires_grad

def test_unfreeze_encoder_all_trainable():
    model = get_model("resnet18", "unet", 7, 2, pretrained="none", freeze_encoder=False)
    for param in model.encoder.parameters():
        assert param.requires_grad

def test_frozen_encoder_decoder_still_trains():
    """Decoder grads should flow even when encoder is frozen."""
    model = get_model("resnet18", "unet", 7, 2, pretrained="none", freeze_encoder=True)
    loss = model(torch.randn(1, 7, 128, 128)).mean()
    loss.backward()
    # Decoder conv blocks should have gradients
    for param in model.dec4.parameters():
        assert param.grad is not None


# ===================================================================
# _adapt_first_conv weight mapping
# ===================================================================

def _make_backbone(in_channels):
    """Minimal mock backbone with a conv1 attribute."""
    backbone = types.SimpleNamespace()
    backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, bias=False)
    return backbone

def test_adapt_first_conv_output_shape():
    backbone = _make_backbone(3)
    backbone = _adapt_first_conv(backbone, 7, "imagenet")
    assert backbone.conv1.weight.shape == (64, 7, 7, 7)

def test_adapt_first_conv_imagenet_rgb_channel_mapping():
    """Blue input channel (idx 0) should receive the Red filter (idx 2) from ImageNet."""
    backbone = _make_backbone(3)
    original_weights = backbone.conv1.weight.data.clone()
    backbone = _adapt_first_conv(backbone, 3, "imagenet")
    assert torch.allclose(backbone.conv1.weight[:, 0], original_weights[:, 2])  # B ← R
    assert torch.allclose(backbone.conv1.weight[:, 1], original_weights[:, 1])  # G ← G
    assert torch.allclose(backbone.conv1.weight[:, 2], original_weights[:, 0])  # R ← B

def test_adapt_first_conv_imagenet_extra_channels_use_mean():
    """Channels beyond 3 should be initialised to the mean of original RGB filters."""
    backbone = _make_backbone(3)
    original_weights = backbone.conv1.weight.data.clone()
    backbone = _adapt_first_conv(backbone, 5, "imagenet")
    expected = original_weights.mean(dim=1)
    assert torch.allclose(backbone.conv1.weight[:, 3], expected)
    assert torch.allclose(backbone.conv1.weight[:, 4], expected)

def test_adapt_first_conv_bigearthnet_band_mapping():
    """First 4 channels should map to B2, B3, B4, B8 from the 12-band BigEarthNet conv."""
    backbone = _make_backbone(12)
    original_weights = backbone.conv1.weight.data.clone()
    backbone = _adapt_first_conv(backbone, 4, "bigearthnet")
    ben_map = [1, 2, 3, 7]  # B2, B3, B4, B8
    for i, src in enumerate(ben_map):
        assert torch.allclose(backbone.conv1.weight[:, i], original_weights[:, src])

def test_adapt_first_conv_none_preserves_shape():
    """pretrained='none' should just reshape conv1 without touching weights logic."""
    backbone = _make_backbone(3)
    backbone = _adapt_first_conv(backbone, 7, "none")
    assert backbone.conv1.weight.shape == (64, 7, 7, 7)


# ===================================================================
# Miscellaneous
# ===================================================================

def test_model_speed():
    model = unet()
    with torch.no_grad():
        model(torch.randn(1, 3, 128, 128))  # should complete without error on CPU

def test_parameter_count_scratch():
    model = unet()
    assert sum(p.numel() for p in model.parameters()) > 1e6

def test_parameter_count_resnet():
    model = get_model("resnet18", "unet", 7, 2, pretrained="none")
    assert sum(p.numel() for p in model.parameters()) > 1e6
