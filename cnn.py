import torch
import torch.nn as tnn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential

def conv_layer(channel_in, channel_out, kernel_size, stride, padding_size = 0):
    layer = Sequential(
        tnn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding_size),
        tnn.BatchNorm2d(channel_out),
        tnn.ReLU()
    )
    return layer

def conv_group(channel_in, channel_out_list, conv_kernel_list, conv_stride_list):
    layers = [ conv_layer(channel_in, channel_out_list[0], conv_kernel_list[0], conv_stride_list[0]) ]
    layers += [ conv_layer(channel_out_list[i], channel_out_list[i + 1], conv_kernel_list[i + 1], conv_stride_list[i + 1])
                for i in range(len(channel_out_list) - 1) ]
    return Sequential(*layers, tnn.MaxPool2d(kernel_size=2))

def fc_layer(dim_in, dim_out):
    layer = Sequential(
        tnn.Linear(dim_in, dim_out),
        tnn.BatchNorm1d(dim_out),
        tnn.ReLU()
    )
    return layer

class cnn(tnn.Module):
    def __init__(self, n_classes):
        super(cnn, self).__init__()
        # input size 200 * 200 * 3 --ConvGroups--> 5 * 5 * 512
        # conv layers
        self.layer0 = conv_group(3,   [16, 16, 16, 32],     [3] * 4, [1] * 4) # 4 conv + 1 max pool: (200-2-2-2-2)/2 = 96
        self.layer1 = conv_group(32,  [32, 32, 64, 64],     [3] * 4, [1] * 4) # 4 conv + 1 max pool: (96-2-2-2-2)/2  = 44
        self.layer2 = conv_group(64,  [64, 128, 128, 128],  [3] * 4, [1] * 4) # 4 conv + 1 max pool: (44-2-2-2-2)/2  = 18
        self.layer3 = conv_group(128, [256, 256, 256, 512], [3] * 4, [1] * 4) # 4 conv + 1 max pool: (18-2-2-2-2)/2  = 5
        # 16 conv + 4 max pool

        # fc layers
        self.layer4 = fc_layer(5 * 5 * 512, 1024)
        self.layer5 = fc_layer(1024, 1024)
        self.layer6 = fc_layer(1024, 1024)
        # output layer
        self.layer7 = tnn.Linear(1024, n_classes)

    def forward(self, x):
        # conv layers
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        # fc layers
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out

