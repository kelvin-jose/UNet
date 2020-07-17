import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True))


def crop_and_copy(input, target):
    input_size = input.shape[2]
    target_size = target.shape[2]
    diff = (input_size - target_size) // 2
    return input[:, :, diff:input_size - diff, diff:input_size - diff]


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1 = double_conv(1, 64)
        self.conv_2 = double_conv(64, 128)
        self.conv_3 = double_conv(128, 256)
        self.conv_4 = double_conv(256, 512)
        self.conv_5 = double_conv(512, 1024)
        self.tconv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)
        self.tconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)
        self.tconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)
        self.tconv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, image):
        x1 = self.conv_1(image)
        x1_max = self.max_pool(x1)
        x2 = self.conv_2(x1_max)
        x2_max = self.max_pool(x2)
        x3 = self.conv_3(x2_max)
        x3_max = self.max_pool(x3)
        x4 = self.conv_4(x3_max)
        x4_max = self.max_pool(x4)
        x5 = self.conv_5(x4_max)
        x6 = self.tconv_1(x5)
        x6_cat = torch.cat((crop_and_copy(x4, x6), x6), 1)
        x7 = self.up_conv_1(x6_cat)
        x8 = self.tconv_2(x7)
        x8_cat = torch.cat((crop_and_copy(x3, x8), x8), 1)
        x9 = self.up_conv_2(x8_cat)
        x10 = self.tconv_3(x9)
        x10_cat = torch.cat((crop_and_copy(x2, x10), x10), 1)
        x11 = self.up_conv_3(x10_cat)
        x12 = self.tconv_4(x11)
        x12_cat = torch.cat((crop_and_copy(x1, x12), x12), 1)
        x13 = self.up_conv_4(x12_cat)
        logits = self.final(x13)
        return logits


if __name__ == "__main__":
    unet = UNet()
    image = torch.rand(1, 1, 572, 572)
    logits = unet(image)
    print(logits)
