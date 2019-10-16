# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, outplanes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, planes, block=Bottleneck, layers=[2, 3, 3, 3], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.out_dim = planes[-1]
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, planes[0], planes[1], layers[0])
        self.layer2 = self._make_layer(block, planes[1], planes[2], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, planes[2], planes[3], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, planes[3], planes[4], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=0)
        self.fc = nn.Conv2d(
            planes[-1], planes[-1], kernel_size=(1, 2), stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, outplanes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, outplanes, stride),
                norm_layer(outplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, outplanes, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, outplanes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x.shape = (batch, channel, time, frequency)
        # in_x  = (batch, 1, 250(2.5sec), 257(fft point))
        # out_x = (batch, last_layer.outplanes, time/32, 1)
        if len(x.shape) <= 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.maxpool2(x)
        x = self.fc(x)
        x = self.relu(x)

        return x


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=8, dim=512, alpha=1.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -
                                  1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class ThinResNet(nn.Module):
    def __init__(self, speaker_num, time_dim, loss_fn, spkr_dim, resnet_config, netvlad_config):
        super(ThinResNet, self).__init__()
        self.resnet = ResNet(**resnet_config)
        self.netvlad = NetVLAD(dim=self.resnet.out_dim, **netvlad_config)
        self.time_dim = time_dim
        #vlad_dim = (time_dim + 31) // 32 * self.resnet.out_dim
        vlad_dim = self.netvlad.num_clusters * self.resnet.out_dim
        self.fc = nn.Linear(vlad_dim, spkr_dim)
        self.prediction_layer = nn.Linear(spkr_dim, speaker_num, bias=False)
        self.loss_fn = loss_fn

    def forward(self, x, hidden_len, rand_cut=True):
        x_cut = x[:, :self.time_dim, :]
        # Cut input feature to fixed size(self.time_dim)
        for i, cut_end in enumerate(hidden_len):
            if rand_cut:
                rand_end = cut_end - self.time_dim
                rand_end = rand_end if rand_end > 0 else 1
                cut_start = np.random.random_integers(0, rand_end)
            else:
                cut_start = 0
            x_cut[i] = x[i, cut_start:cut_start+self.time_dim]
        extracted_feature = self.resnet(x_cut)
        vlad = self.netvlad(extracted_feature)
        speaker_vector = self.fc(vlad)
        if self.loss_fn == 'softmax':
            y_pred = self.prediction_layer(speaker_vector)
            y_pred = F.softmax(y_pred, dim=1)
        elif self.loss_fn == 'amsoftmax':
            speaker_vector = F.normalize(speaker_vector, p=2, dim=1)
            y_pred = self.prediction_layer(speaker_vector)
        else:
            raise NotImplementedError

        return speaker_vector, y_pred
