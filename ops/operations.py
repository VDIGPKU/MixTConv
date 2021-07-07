# operations for MixTConv
# arXiv:TBD
# Kaiyu Shan
# shankyle@pku.edu.cn
# ------------------------------------------------------
# Code adapted from https://github.com/mit-han-lab/temporal-shift-module
import torch
import torch.nn as nn
import torch.nn.functional as F
# from arch import *
# from spatial_correlation_sampler import SpatialCorrelationSampler
import math
from IPython import embed


class MsGroupConv1d(nn.Module):
    def __init__(self, net, n_segment=8, n_div=4, inplace=True):
        super(MsGroupConv1d, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.inplace = inplace
        if inplace:
            print('=> Using in-place multi-scale 1d conv...')
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv11d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv13d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv15d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv17d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=net.in_channels // 4)
        self.net = net
        self.weight_init()

    def ms_groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x1, x3, x5, x7 = x.split([c // 4, c // 4, c // 4, c // 4], dim=1)

        x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)

        x = torch.cat((x1, x3, x5, x7), dim=1).view(nt, c, h, w)

        return x

    def weight_init(self):
        print('=> Using weight init of 4 parts for 4 multi-scale')
        planes = self.conv11d.in_channels
        fold = planes // self.fold_div # div = 4

        weight1 = torch.zeros(planes, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
        weight1[:, 0, 0] = 1.0
        self.conv11d.weight = nn.Parameter(weight1)

        # diff 1357 = shift + stride 0 2 4 6
        weight3 = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W] [010]:1/2 [100]1/4 [110]1/4
        weight3[:fold, 0, 0] = 1.0
        weight3[fold: fold * 2, 0, 2] = 1.0
        weight3[fold * 2:, 0, 1] = 1.0
        self.conv13d.weight = nn.Parameter(weight3)

        weight5 = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight5[:fold, 0, :2] = 1.0  # [11000]
        weight5[fold: fold * 2, 0, 3:] = 1.0  # [00011]
        weight5[fold * 2:, 0, 2] = 1.0 # [00100]
        self.conv15d.weight = nn.Parameter(weight5)

        weight7 = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight7[:fold, 0, :3] = 1.0  # [1110000]
        weight7[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
        weight7[fold * 2:, 0, 3] = 1.0 # [0001000]
        self.conv17d.weight = nn.Parameter(weight7)

    def forward(self, x):
        x = self.ms_groupconv1d(x)
        return self.net(x)


def make_operations(net, n_segment, n_div=8, operations='baseline', dwise=False, corr_group=1, inplace=False):
    if operations == 'baseline':
        print('=> Using Operations as {}'.format(operations))
        import torchvision

    if operations == 'ms_group1douter':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = MsGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div, inplace=False)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    else:
        raise NotImplementedError(operations)


if __name__ == '__main__':
    import torchvision
    model = torchvision.models.resnet50(True)
    make_operations(model, 8, n_div=8, operations='msgroup1dpartalsplitdiv')
    data = torch.autograd.Variable(torch.ones(16, 3, 320, 256))
    out = model(data)
    out.mean().backward()
    print(model)
    print(out.size())




