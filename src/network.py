# Image segmentation networks implemented in PyTorch
# Supported architectures: U-Net, R2U-Net, AttU-Net, R2AttU-Net, SWED-UNet
# Encoders: scratch, resnet18, resnet50, resnet101 (ImageNet or BigEarthNet pretrained)

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ---------------------------------------------------------
# ResNet encoder configuration
# ---------------------------------------------------------

BIGEARTHNET_IDS = {
    "resnet18":  "BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0",
    "resnet50":  "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
    "resnet101": "BIFOLD-BigEarthNetv2-0/resnet101-s2-v0.2.0",
}

# Output channels at each encoder stage: [act1, layer1, layer2, layer3, layer4]
ENCODER_CHANNELS = {
    "resnet18":  [64,   64,  128,  256,  512],
    "resnet50":  [64,  256,  512, 1024, 2048],
    "resnet101": [64,  256,  512, 1024, 2048],
}


# ---------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------

def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


# ---------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            recurrent_block(ch_out, t=t), recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ---------------------------------------------------------
# Scratch encoder U-Net variants
# ---------------------------------------------------------

class unet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(unet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=input_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))
        d5 = self.Up_conv5(torch.cat((x4, self.Up5(x5)), dim=1))
        d4 = self.Up_conv4(torch.cat((x3, self.Up4(d5)), dim=1))
        d3 = self.Up_conv3(torch.cat((x2, self.Up3(d4)), dim=1))
        d2 = self.Up_conv2(torch.cat((x1, self.Up2(d3)), dim=1))
        return self.Conv_1x1(d2)


class r2_unet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, t=2):
        super(r2_unet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.RRCNN1 = RRCNN_block(ch_in=input_channels, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)
        self.Conv_1x1 = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.RRCNN1(x)
        x2 = self.RRCNN2(self.Maxpool(x1))
        x3 = self.RRCNN3(self.Maxpool(x2))
        x4 = self.RRCNN4(self.Maxpool(x3))
        x5 = self.RRCNN5(self.Maxpool(x4))
        d5 = self.Up_RRCNN5(torch.cat((x4, self.Up5(x5)), dim=1))
        d4 = self.Up_RRCNN4(torch.cat((x3, self.Up4(d5)), dim=1))
        d3 = self.Up_RRCNN3(torch.cat((x2, self.Up3(d4)), dim=1))
        d2 = self.Up_RRCNN2(torch.cat((x1, self.Up2(d3)), dim=1))
        return self.Conv_1x1(d2)


class att_unet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(att_unet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=input_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))
        return self.Conv_1x1(d2)


class r2att_unet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, t=2):
        super(r2att_unet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.RRCNN1 = RRCNN_block(ch_in=input_channels, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)
        self.Conv_1x1 = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.RRCNN1(x)
        x2 = self.RRCNN2(self.Maxpool(x1))
        x3 = self.RRCNN3(self.Maxpool(x2))
        x4 = self.RRCNN4(self.Maxpool(x3))
        x5 = self.RRCNN5(self.Maxpool(x4))
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = self.Up_RRCNN5(torch.cat((x4, d5), dim=1))
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = self.Up_RRCNN4(torch.cat((x3, d4), dim=1))
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = self.Up_RRCNN3(torch.cat((x2, d3), dim=1))
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = self.Up_RRCNN2(torch.cat((x1, d2), dim=1))
        return self.Conv_1x1(d2)


# ---------------------------------------------------------
# SWED U-Net (Seale et al. 2022)
# Smaller channel widths (32→64→128→256, bottleneck 512),
# ELU activation, Conv→ELU→BN order, ConvTranspose2d upsampling.
# ---------------------------------------------------------

class swed_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(swed_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        return self.conv(x)


class swed_unet(nn.Module):
    def __init__(self, input_channels=12, output_channels=2):
        super(swed_unet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = swed_conv_block(ch_in=input_channels, ch_out=32)
        self.Conv2 = swed_conv_block(ch_in=32,  ch_out=64)
        self.Conv3 = swed_conv_block(ch_in=64,  ch_out=128)
        self.Conv4 = swed_conv_block(ch_in=128, ch_out=256)
        self.Conv5 = swed_conv_block(ch_in=256, ch_out=512)

        self.Up4     = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Up_conv4 = swed_conv_block(ch_in=512, ch_out=256)
        self.Up3     = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Up_conv3 = swed_conv_block(ch_in=256, ch_out=128)
        self.Up2     = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.Up_conv2 = swed_conv_block(ch_in=128, ch_out=64)
        self.Up1     = nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2)
        self.Up_conv1 = swed_conv_block(ch_in=64,  ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        d4 = self.Up_conv4(torch.cat((x4, self.Up4(x5)), dim=1))
        d3 = self.Up_conv3(torch.cat((x3, self.Up3(d4)), dim=1))
        d2 = self.Up_conv2(torch.cat((x2, self.Up2(d3)), dim=1))
        d1 = self.Up_conv1(torch.cat((x1, self.Up1(d2)), dim=1))
        return self.Conv_1x1(d1)


# ---------------------------------------------------------
# ResNet encoder
# ---------------------------------------------------------

def _build_resnet_backbone(encoder_name, pretrained):
    """Load a ResNet backbone with optional pretrained weights."""
    from torchvision.models import (
        resnet18, resnet50, resnet101,
        ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    )

    fn_map = {"resnet18": resnet18, "resnet50": resnet50, "resnet101": resnet101}

    if pretrained == "imagenet":
        w_map = {
            "resnet18": ResNet18_Weights.DEFAULT,
            "resnet50": ResNet50_Weights.DEFAULT,
            "resnet101": ResNet101_Weights.DEFAULT,
        }
        return fn_map[encoder_name](weights=w_map[encoder_name])

    elif pretrained == "bigearthnet":
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = BigEarthNetv2_0_ImageClassifier.from_pretrained(BIGEARTHNET_IDS[encoder_name])
        return clf.model.vision_encoder

    else:  # none — random init
        return fn_map[encoder_name](weights=None)


def _adapt_first_conv(backbone, in_channels, pretrained):
    """Replace conv1 to accept in_channels, transferring pretrained weights where possible."""
    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if pretrained == "imagenet":
            # Pretrained on RGB; input assumed to be ordered B, G, R, [extras]
            channel_map = [2, 1, 0]  # B→R, G→G, R→B (reverse BGR to RGB)
            for i in range(min(in_channels, 3)):
                new_conv.weight[:, i] = old_conv.weight[:, channel_map[i]]
            for i in range(3, in_channels):
                new_conv.weight[:, i] = old_conv.weight.mean(dim=1)

        elif pretrained == "bigearthnet":
            # Pretrained on 12 Sentinel-2 bands ordered:
            # idx: 0=B1, 1=B2(Blue), 2=B3(Green), 3=B4(Red), 4=B5, 5=B6,
            #      6=B7, 7=B8(NIR), 8=B8A, 9=B9, 10=B10, 11=B11
            ben_band_map = [1, 2, 3, 7]  # Blue, Green, Red, NIR
            for i in range(min(in_channels, 4)):
                new_conv.weight[:, i] = old_conv.weight[:, ben_band_map[i]]
            for i in range(4, in_channels):
                new_conv.weight[:, i] = old_conv.weight.mean(dim=1)
        # pretrained == "none": new_conv has default random init, leave as-is

    backbone.conv1 = new_conv
    return backbone


class ResNetEncoder(nn.Module):
    """Wraps a ResNet backbone, extracting 5 feature maps for U-Net skip connections."""

    def __init__(self, encoder_name, pretrained, in_channels, freeze_encoder):
        super().__init__()

        backbone = _build_resnet_backbone(encoder_name, pretrained)
        backbone = _adapt_first_conv(backbone, in_channels, pretrained)

        if freeze_encoder:
            for param in backbone.parameters():
                param.requires_grad = False
            for param in backbone.conv1.parameters():
                param.requires_grad = True

        # Split into named stages for clean skip-connection access
        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, nn.ReLU(inplace=True))  # H/2
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # H/4
        self.layer2 = backbone.layer2  # H/8
        self.layer3 = backbone.layer3  # H/16
        self.layer4 = backbone.layer4  # H/32

    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


# ---------------------------------------------------------
# ResNet-encoder U-Net decoder (standard + attention)
# ---------------------------------------------------------

class ResNetUNet(nn.Module):
    """U-Net with a pretrained ResNet encoder. Supports standard and attention decoders."""

    def __init__(self, encoder_name, pretrained, in_channels, output_channels,
                 freeze_encoder=False, use_attention=False):
        super().__init__()

        self.encoder = ResNetEncoder(encoder_name, pretrained, in_channels, freeze_encoder)
        enc = ENCODER_CHANNELS[encoder_name]  # [c0, c1, c2, c3, c4]

        # Decoder: upsample bottleneck → concat with skip → conv_block
        self.up4 = up_conv(enc[4], enc[3])
        self.dec4 = conv_block(enc[3] * 2, enc[3])

        self.up3 = up_conv(enc[3], enc[2])
        self.dec3 = conv_block(enc[2] * 2, enc[2])

        self.up2 = up_conv(enc[2], enc[1])
        self.dec2 = conv_block(enc[1] * 2, enc[1])

        self.up1 = up_conv(enc[1], enc[0])
        self.dec1 = conv_block(enc[0] * 2, enc[0])

        # Final upsample back to full resolution (stage0 halved it; no skip available)
        self.up0 = up_conv(enc[0], 32)
        self.dec0 = conv_block(32, 32)

        self.Conv_1x1 = nn.Conv2d(32, output_channels, kernel_size=1)

        self.use_attention = use_attention
        if use_attention:
            self.att4 = attention_block(F_g=enc[3], F_l=enc[3], F_int=enc[3] // 2)
            self.att3 = attention_block(F_g=enc[2], F_l=enc[2], F_int=enc[2] // 2)
            self.att2 = attention_block(F_g=enc[1], F_l=enc[1], F_int=enc[1] // 2)
            self.att1 = attention_block(F_g=enc[0], F_l=enc[0], F_int=enc[0] // 2)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        d4 = self.up4(x4)
        if self.use_attention:
            x3 = self.att4(g=d4, x=x3)
        d4 = self.dec4(torch.cat([x3, d4], dim=1))

        d3 = self.up3(d4)
        if self.use_attention:
            x2 = self.att3(g=d3, x=x2)
        d3 = self.dec3(torch.cat([x2, d3], dim=1))

        d2 = self.up2(d3)
        if self.use_attention:
            x1 = self.att2(g=d2, x=x1)
        d2 = self.dec2(torch.cat([x1, d2], dim=1))

        d1 = self.up1(d2)
        if self.use_attention:
            x0 = self.att1(g=d1, x=x0)
        d1 = self.dec1(torch.cat([x0, d1], dim=1))

        d0 = self.dec0(self.up0(d1))
        return self.Conv_1x1(d0)


# ---------------------------------------------------------
# Model factory
# ---------------------------------------------------------

def get_model(encoder, model_type, in_channels, output_channels,
              pretrained="none", freeze_encoder=False, weight_init="normal"):
    """
    Build and return a segmentation model.

    encoder     : "scratch" | "resnet18" | "resnet50" | "resnet101"
    model_type  : "unet" | "r2_unet" | "att_unet" | "r2att_unet"
    pretrained  : "none" | "imagenet" | "bigearthnet"  (ignored for scratch)
    freeze_encoder : freeze ResNet weights except modified conv1
    weight_init : init scheme for scratch encoder
    """
    if encoder == "scratch":
        if pretrained != "none":
            raise ValueError(
                f"pretrained='{pretrained}' requires a ResNet encoder, not 'scratch'."
            )
        scratch_map = {
            "unet":       unet,
            "r2_unet":    r2_unet,
            "att_unet":   att_unet,
            "r2att_unet": r2att_unet,
            "swed_unet":  swed_unet,
        }
        model = scratch_map[model_type](in_channels, output_channels)
        init_weights(model, weight_init)
        return model

    # ResNet encoder path
    if model_type in ("r2_unet", "r2att_unet", "swed_unet"):
        raise ValueError(
            f"model_type='{model_type}' is not compatible with a pretrained ResNet encoder. "
            f"Use 'unet' or 'att_unet' with encoder='{encoder}'."
        )

    use_attention = (model_type == "att_unet")
    return ResNetUNet(
        encoder_name=encoder,
        pretrained=pretrained,
        in_channels=in_channels,
        output_channels=output_channels,
        freeze_encoder=freeze_encoder,
        use_attention=use_attention,
    )


def load_model(model_name, models_dir, device=None):
    """
    Load a saved model by name and return (model, config).

    model_name : filename stem, e.g. "LICS_unet_adam"
    models_dir : directory containing the .pth and .json files
    device     : torch device string/object; defaults to cuda > mps > cpu
    """
    import json, os

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    json_path = os.path.join(models_dir, f"{model_name}.json")
    pth_path  = os.path.join(models_dir, f"{model_name}.pth")

    with open(json_path) as f:
        config = json.load(f)

    in_channels  = len(config["incl_bands"])
    out_channels = 1 if config.get("binary_mask", False) else 2
    # Use pretrained="none" to avoid re-downloading backbone weights —
    # the .pth already contains all final weights which are loaded below.
    model = get_model(
        encoder         = config["encoder"],
        model_type      = config["model_type"],
        in_channels     = in_channels,
        output_channels = out_channels,
        pretrained      = "none",
        freeze_encoder  = False,
        weight_init     = config.get("weight_init", "normal"),
    )

    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, config
