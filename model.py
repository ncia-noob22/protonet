import torch.nn as nn


class ProtoNet(nn.Module):
    def __init__(self, input_dim=1, hid_dim=64, z_dim=64):
        super().__init__()
        self.block1 = ConvBlock(
            input_dim, hid_dim, kernel_size=3, max_pool=2, padding=1
        )
        self.block2 = ConvBlock(hid_dim, hid_dim, kernel_size=3, max_pool=2, padding=1)
        self.block3 = ConvBlock(hid_dim, hid_dim, kernel_size=3, max_pool=2, padding=1)
        self.block4 = ConvBlock(hid_dim, z_dim, kernel_size=3, max_pool=2, padding=1)

    def forward(self, x):
        out = self.block4(self.block3(self.block2(self.block1(x)))).flatten(start_dim=1)
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        activation="relu",
        use_bn=True,
        max_pool=None,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.use_max_pool = max_pool is not None

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

        if activation.lower() == "relu":
            self.activ = nn.ReLU(inplace=True)
        elif activation.lower() == "relu6":
            self.activ = nn.ReLU6(inplace=True)
        elif activation.lower() == "sigmoid":
            self.activ = nn.Sigmoid()

        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.use_max_pool:
            self.max_pool = nn.MaxPool2d(max_pool)

    def forward(self, x):
        x = self.activ(self.conv(x))
        if self.use_bn:
            x = self.bn(x)
        if self.use_max_pool:
            x = self.max_pool(x)
        return x
