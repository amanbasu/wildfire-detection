import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding='same', dilation=dilation)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding='same', dilation=dilation)
        self.bnorm1 = nn.BatchNorm2d(channel_out)
        self.bnorm2 = nn.BatchNorm2d(channel_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.activation(self.bnorm1(conv1))
        conv2 = self.conv2(conv1)
        conv2 = self.activation(self.bnorm2(conv2))
        return conv2

class Downsample(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.downsample = nn.Conv2d(channel_in, channel_in * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv_transpose(x)

class UNet(nn.Module):
    def __init__(self, clannels, classes):
        super(UNet, self).__init__()
        self.CHANNELS = clannels
        self.CLASSES = classes

        self.inp = ConvBlock(self.CHANNELS, 64)

        self.stage1 = ConvBlock(128, 128, dilation=1)
        self.stage2 = ConvBlock(256, 256, dilation=1)
        self.stage3 = ConvBlock(512, 512, dilation=2)
        self.stage4 = ConvBlock(1024, 1024, dilation=3)

        self.down1 = Downsample(64)
        self.down2 = Downsample(128)
        self.down3 = Downsample(256)
        self.down4 = Downsample(512)

        self.up1 = Upsample(1024, 512)
        self.up2 = Upsample(512, 256)
        self.up3 = Upsample(256, 128)
        self.up4 = Upsample(128, 64)

        self.stage4i = ConvBlock(1024, 512, dilation=3)
        self.stage3i = ConvBlock(512, 256, dilation=2)
        self.stage2i = ConvBlock(256, 128, dilation=1)
        self.stage1i = ConvBlock(128, 64, dilation=1)

        self.out = nn.Conv2d(64, self.CLASSES, kernel_size=1)

    def forward(self, x):
        a1 = self.inp(x)
        d1 = self.down1(a1)

        a2 = self.stage1(d1)
        d2 = self.down2(a2)

        a3 = self.stage2(d2)
        d3 = self.down3(a3)

        a4 = self.stage3(d3)
        d4 = self.down4(a4)

        a5 = self.stage4(d4)
        u1 = self.up1(a5)

        c1 = self.stage4i(torch.cat([a4, u1], dim=1))
        u2 = self.up2(c1)

        c2 = self.stage3i(torch.cat([a3, u2], dim=1))
        u3 = self.up3(c2)

        c3 = self.stage2i(torch.cat([a2, u3], dim=1))
        u4 = self.up4(c3)

        c4 = self.stage1i(torch.cat([a1, u4], dim=1))
        logits = self.out(c4)

        return logits