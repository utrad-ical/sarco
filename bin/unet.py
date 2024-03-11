


import numpy as np
import torch
import torch.nn as nn







class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock2D, self).__init__()
        
        self.gn1 = nn.GroupNorm(16, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(16, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels == out_channels:
            self.shortcutfunc = lambda x:x
        else:
            self.shortcutfunc = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        shortcut = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + self.shortcutfunc(shortcut)
        return x



class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv = ResnetBlock2D(self.in_channels,
                                  self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pooling:
            x = self.pool(x)
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv = ResnetBlock2D(self.in_channels,
                                  self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upsample(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = self.conv(x)
        return x



class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, start_filter_num=16):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.start_filter_num = start_filter_num

        self.conv_initial = nn.Conv2d(in_channels, start_filter_num, 1)

        self.down_conv1 = DownConv(start_filter_num, start_filter_num, pooling=False)
        self.down_conv2 = DownConv(start_filter_num*1, start_filter_num*2)
        self.down_conv3 = DownConv(start_filter_num*2, start_filter_num*4)
        self.down_conv4 = DownConv(start_filter_num*4, start_filter_num*8)
        self.down_conv5 = DownConv(start_filter_num*8, start_filter_num*8)
        self.down_conv6 = DownConv(start_filter_num*8, start_filter_num*8)
        self.down_conv7 = DownConv(start_filter_num*8, start_filter_num*8)
        self.down_conv8 = DownConv(start_filter_num*8, start_filter_num*8)
        self.down_conv9 = DownConv(start_filter_num*8, start_filter_num*8)
        self.down_conv10 = DownConv(start_filter_num*8, start_filter_num*8)

        self.fc = nn.Linear(start_filter_num*8, start_filter_num*8)

        self.up_conv9   = UpConv(start_filter_num*16, start_filter_num* 8)
        self.up_conv8   = UpConv(start_filter_num*16, start_filter_num* 8)
        self.up_conv7   = UpConv(start_filter_num*16, start_filter_num* 8)
        self.up_conv6   = UpConv(start_filter_num*16, start_filter_num* 8)
        self.up_conv5   = UpConv(start_filter_num*16, start_filter_num* 8)
        self.up_conv4   = UpConv(start_filter_num*16, start_filter_num* 8)
        self.up_conv3   = UpConv(start_filter_num*12, start_filter_num* 4)
        self.up_conv2   = UpConv(start_filter_num* 6, start_filter_num* 2)
        self.up_conv1   = UpConv(start_filter_num* 3, start_filter_num* 1)

        self.conv_final = nn.Conv2d(start_filter_num, out_channels, 1)
        

    def forward(self, x):
        x = self.conv_initial(x)
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)
        x6 = self.down_conv6(x5)
        x7 = self.down_conv7(x6)
        x8 = self.down_conv8(x7)
        x9 = self.down_conv9(x8)
        x10 = self.down_conv10(x9)

#        x10 = torch.reshape(self.fc(torch.reshape(x10, [-1,128])), [-1,128,1,1])

        x = self.up_conv9(x9, x10)
        x = self.up_conv8(x8, x)
        x = self.up_conv7(x7, x)
        x = self.up_conv6(x6, x)
        x = self.up_conv5(x5, x)
        x = self.up_conv4(x4, x)
        x = self.up_conv3(x3, x)
        x = self.up_conv2(x2, x)
        x = self.up_conv1(x1, x)
        
        return self.conv_final(x)
        