# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.checkpoint as cp
import sys
from inplace_abn import InPlaceABN, InPlaceABNSync
sys.path.insert(0,'../')
import config
#####################################################################################
# densenet
# if config.multi_gpu and not(config.no_DDP_thistime):
#     Batchnorm = InPlaceABNSync
# else:
Batchnorm = InPlaceABN

def conv3d_ABN(ni, nf, stride, activation="relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv3d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        Batchnorm(num_features=nf, activation=activation, activation_param=activation_param)
    )


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
        #                                    growth_rate, kernel_size=1, stride=1, bias=False)),
        # self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
        #                                    kernel_size=3, stride=1, padding=1, bias=False)),

        self.bn1 = Batchnorm(num_features=num_input_features, activation="leaky_relu")
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = Batchnorm(num_features=bn_size * growth_rate, activation="leaky_relu")
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, x):
        # bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        # if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
        #     bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        # else:
        #     bottleneck_output = bn_function(*prev_features)
        # new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        # if self.drop_rate > 0:
        #     new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        concated_features = torch.cat(x, 1)
        out = self.bn1(concated_features)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        return out


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.add_module('norm', nn.BatchNorm3d(num_input_features))
        # self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
        #                                   kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

        self.norm = Batchnorm(num_features=num_input_features, activation="leaky_relu")
        self.conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
    def forward(self, prev_features):
        out = self.norm(prev_features)
        out = self.conv(out)
        out = self.pool(out)
        return out

class Densenet36(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, k1=15, k2=30):
        super(Densenet36, self).__init__()


        # First convolution
        # self.features_block1 = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('norm0', nn.BatchNorm3d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True))
        # ]))

        self.conv0 = nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm0 =  Batchnorm(num_features = num_init_features, activation="leaky_relu")
        self.features_block1 = nn.Sequential(OrderedDict([]))
        # Each denseblock
        num_features = num_init_features

        # block1
        i = 0
        num_layers = block_config[i]
        block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features_block1.add_module('denseblock%d' % (i + 1), block)
        num_features = num_features + num_layers * growth_rate
        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features_block1.add_module('transition%d' % (i + 1), trans)
        num_features = num_features // 2
        # block2
        i = 1
        num_layers = block_config[i]
        self.features_block2 = nn.Sequential(OrderedDict([]))
        block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features_block2.add_module('denseblock%d' % (i + 1), block)
        num_features = num_features + num_layers * growth_rate
        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features_block2.add_module('transition%d' % (i + 1), trans)
        num_features = num_features // 2
        num_features_b2 = num_features
        # block3
        i = 2
        num_layers = block_config[i]
        self.features_block3 = nn.Sequential(OrderedDict([]))
        block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features_block3.add_module('denseblock%d' % (i + 1), block)
        num_features = num_features + num_layers * growth_rate
        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features_block3.add_module('transition%d' % (i + 1), trans)
        num_features = num_features // 2

        # Final batch norm
        self.features_block3.add_module('norm5', nn.BatchNorm3d(num_features))
        # num_features = 6*6*6*508
        # Linear layer
        self.num_classes = num_classes
        self.k1 = k1
        self.k2 = k2

        # G stream
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features

        # P stream

        self.conv_P = torch.nn.Conv3d(num_features, k1 * num_classes, kernel_size=1, stride=1, padding=0)
        # self.pool6 = pool6
        self.cls_P = nn.Linear(k1 * num_classes, num_classes)
        # Side-branch
        self.cross_channel_pool = nn.AvgPool1d(kernel_size=k1, stride=k1, padding=0)

        # dfl_early
        self.num_features_b2 = num_features_b2 + k1 * num_classes

        # P stream
        self.conv_P_b2_pre = _DenseBlock(num_layers=6, num_input_features=self.num_features_b2,
                                         bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_features_b2 = self.num_features_b2 + 6 * growth_rate

        self.conv_P_b2 = torch.nn.Conv3d(self.num_features_b2, k2 * num_classes, kernel_size=1, stride=1, padding=0)
        # self.pool6 = pool6
        self.cls_P_b2 = nn.Linear(k2 * num_classes, num_classes)
        # Side-branch
        self.cross_channel_pool_b2 = nn.AvgPool1d(kernel_size=k2, stride=k2, padding=0)

    def forward(self, x):
        # backbone

        out = self.norm0(self.conv0(x))
        out = self.features_block1(out)
        out = self.features_block2(out)
        out_b2 = out
        out = self.features_block3(out)
        # branch G
        out = F.relu(out)
        out_G = F.avg_pool3d(out, out.size(2)).view(x.size(0), -1)
        feat = out_G
        out_G = self.classifier(out_G)

        # branch P & side for low resolution
        out_P_major = self.conv_P(out)
        out_P_major_b1 = out_P_major
        out_P_major, indices = F.max_pool3d(out_P_major, out_P_major.size(2), return_indices=True)
        out_P = self.cls_P(out_P_major.view(x.size(0), -1))

        out_side = out_P_major.view(x.size(0), -1, self.k1 * self.num_classes)
        out_side = self.cross_channel_pool(out_side)
        out_side = out_side.view(x.size(0), self.num_classes)

        # branch P & side for hign resolution
        out_P_major_b1 = F.upsample(out_P_major_b1, scale_factor=2, mode='trilinear')
        out_P_major_b2 = torch.cat((out_b2, out_P_major_b1), dim=1)
        out_P_major_b2 = self.conv_P_b2_pre(out_P_major_b2)
        out_P_major_b2 = self.conv_P_b2(out_P_major_b2)

        out_P_major_b2, indices_b2 = F.max_pool3d(out_P_major_b2, out_P_major_b2.size(2), return_indices=True)
        out_P_b2 = self.cls_P_b2(out_P_major_b2.view(x.size(0), -1))
        out_side_b2 = out_P_major_b2.view(x.size(0), -1, self.k2 * self.num_classes)
        out_side_b2 = self.cross_channel_pool_b2(out_side_b2)
        out_side_b2 = out_side_b2.view(x.size(0), self.num_classes)

        return out_G, out_P, out_side, out_P_b2, out_side_b2, feat

def Densenet36_fgpn(**kwargs):
    model = Densenet36(num_init_features=64, growth_rate=16, block_config=(6, 12, 24), **kwargs)
    return model



