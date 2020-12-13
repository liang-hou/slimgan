import torch.nn as nn
from torch_mimicry.modules import SNConv2d, SNLinear, ConditionalBatchNorm2d


class FLAGS(object):
    width_mult_list = [0.25, 0.5, 0.75, 1.0]
    width_mult = 1.0


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(FLAGS.width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        self.width_mult = FLAGS.width_mult
        idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SwitchableConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features_list, num_classes):
        super(SwitchableConditionalBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        cbns = []
        for i in num_features_list:
            cbns.append(ConditionalBatchNorm2d(i, num_classes))
        self.cbn = nn.ModuleList(cbns)
        self.width_mult = max(FLAGS.width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input, label):
        self.width_mult = FLAGS.width_mult
        idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.cbn[idx](input, label)
        return y


class SliceableConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features_list, num_classes):
        super(SliceableConditionalBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        cbns = []
        for i in num_features_list:
            cbns.append(nn.BatchNorm2d(i))
        self.cbn = nn.ModuleList(cbns)

        self.width_mult = max(FLAGS.width_mult_list)
        self.ignore_model_profiling = True

        self.num_features = max(num_features_list)
        self.embed = nn.Embedding(num_classes, self.num_features * 2)
        self.embed.weight.data[:, :self.num_features].normal_(
            1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, self.num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        self.width_mult = FLAGS.width_mult
        idx = FLAGS.width_mult_list.index(self.width_mult)
        out = self.cbn[idx](x)
        gamma, beta = self.embed(y).chunk(
            2, 1)  # divide into 2 chunks, split from dim 1.
        out = gamma.view(-1, self.num_features, 1, 1)[:, :self.num_features_list[idx], :, :] * out + beta.view(
            -1, self.num_features, 1, 1)[:, :self.num_features_list[idx], :, :]

        return out


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        self.width_mult = FLAGS.width_mult
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConvTranspose2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        self.width_mult = FLAGS.width_mult
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.in_channels, :int(self.out_channels // self.groups), :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv_transpose2d(
            input, weight, bias, self.stride, self.padding,
            dilation=self.dilation, groups=self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        self.width_mult = FLAGS.width_mult
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class SNSlimmableConv2d(SNConv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SNSlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        self.width_mult = FLAGS.width_mult
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.sn_weights()[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SNSlimmableLinear(SNLinear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SNSlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        self.width_mult = FLAGS.width_mult
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.sn_weights()[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)