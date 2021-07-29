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

def conv_group(conv, pooling_sz=4, pooling_stride=1):
    return Sequential(conv, tnn.MaxPool2d(kernel_size=pooling_sz, stride=pooling_stride))

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
        # input size 200 * 200 * 3 -> 100 * 100 * 3
        # conv layers
        self.layer0 = conv_group(conv_layer(3, 16, 5, 1))
        self.layer1 = conv_group(conv_layer(16, 16, 5, 1))
        self.layer2 = conv_group(conv_layer(16, 16, 5, 1))
        self.layer3 = conv_group(conv_layer(16, 16, 5, 1))
        self.layer4 = conv_group(conv_layer(16, 16, 5, 1))
        self.layer5 = conv_group(conv_layer(16, 16, 5, 1))
        self.layer6 = conv_group(conv_layer(16, 32, 5, 1))
        self.layer7 = conv_group(conv_layer(32, 32, 5, 1))
        self.layer8 = conv_group(conv_layer(32, 32, 5, 1))
        self.layer9 = conv_group(conv_layer(32, 32, 5, 1))
        self.layer10 = conv_group(conv_layer(32, 32, 5, 1))
        self.layer11 = conv_group(conv_layer(32, 32, 5, 1))
        self.layer12 = conv_group(conv_layer(32, 64, 5, 1))
        # fc layers
        self.layer13 = fc_layer(9 * 9 * 64, 512)
        self.layer14 = fc_layer(512, 512)
        self.layer15 = fc_layer(512, 512)
        # output layer
        self.layer16 = tnn.Linear(512, n_classes)

    def forward(self, x):
        # conv layers
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = x.view(x.size(0), -1)
        # fc layers
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        return x
