# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from einops import rearrange

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return x


class Corr(nn.Module):
    def __init__(self, in_channels, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(in_channels, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):
        dict_return = {}
        h_in, w_in = feature_in.shape[2], feature_in.shape[3]
        h_out, w_out = out.shape[2], out.shape[3]
        out = torch.nn.functional.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = torch.nn.functional.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = rearrange(out, 'n c h w -> n c (h w)')
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1], device=f1.device).float())
        corr_map = torch.nn.functional.softmax(corr_map, dim=-1)
        corr_map_sample = self.sample(corr_map.detach(), h_in, w_in)
        dict_return['corr_map'] = self.normalize_corr_map(corr_map_sample, h_in, w_in, h_out, w_out)
        dict_return['out'] = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)
        return dict_return

    def sample(self, corr_map, h_in, w_in):
        sample_count = min(128, h_in * w_in)
        index = torch.randint(0, h_in * w_in, [sample_count], device=corr_map.device)
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample

    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m, _ = corr_map.shape
        corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = torch.nn.functional.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)
        corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
        range_ = range_.clamp_min(1e-8)
        temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = (temp_map > 0.5)
        norm_corr_map = rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        return norm_corr_map

def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x

class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.classifier = self.decoder.out_conv
        self.is_corr = True
        self.proj = nn.Sequential(
            nn.Conv2d(params['feature_chns'][-1], params['feature_chns'][0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(params['feature_chns'][0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.corr = Corr(in_channels=params['feature_chns'][0], nclass=class_num)

    def forward(self, x, need_fp=False, use_corr=False, use_feature=False):
        feature = self.encoder(x)
        dict_return = {}

        if need_fp:
            decoder_feature = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            outs = self.classifier(decoder_feature)
            out, out_fp = outs.chunk(2)
            if use_corr:
                proj_feats = self.proj(feature[-1])
                corr_out_dict = self.corr(proj_feats, out)
                dict_return['corr_map'] = corr_out_dict['corr_map']
                dict_return['corr_out'] = torch.nn.functional.interpolate(
                    corr_out_dict['out'],
                    size=out.shape[-2:],
                    mode='bilinear',
                    align_corners=True,
                )
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp
            return dict_return

        decoder_feature = self.decoder(feature)
        out = self.classifier(decoder_feature)
        if use_feature:
            return out, decoder_feature

        if use_corr:
            proj_feats = self.proj(feature[-1])
            corr_out_dict = self.corr(proj_feats, out)
            dict_return['corr_map'] = corr_out_dict['corr_map']
            dict_return['corr_out'] = torch.nn.functional.interpolate(
                corr_out_dict['out'],
                size=out.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
            dict_return['out'] = out
            return dict_return
        return out
