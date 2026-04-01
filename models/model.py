# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import torch, sys, os
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from models.model_process import init_weights

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

class UnetConv3_sigmoid(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3_sigmoid, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.Sigmoid(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.Sigmoid(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.Sigmoid(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.Sigmoid(),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    
class Encoder_sigmoid(nn.Module):
    def __init__(self, params):
        super(Encoder_sigmoid, self).__init__()

        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        
        self.conv1 = UnetConv3_sigmoid(self.in_chns, self.ft_chns[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3_sigmoid(self.ft_chns[0], self.ft_chns[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3_sigmoid(self.ft_chns[1], self.ft_chns[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3_sigmoid(self.ft_chns[2], self.ft_chns[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3_sigmoid(self.ft_chns[3], self.ft_chns[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        
        center = self.center(maxpool4)
        center = self.dropout(center)
        
        return [conv1, conv2, conv3, conv4, center]

class UnetUp3_CT_sigmoid(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT_sigmoid, self).__init__()
        self.conv = UnetConv3_sigmoid(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        self.deep_feature = torch.cat([outputs1, outputs2], 1)
        return self.conv(torch.cat([outputs1, outputs2], 1))
    
class Decoder_wtcls_sigmoid(nn.Module):
    def __init__(self, params, in_channels=3, is_batchnorm=True):
        super(Decoder_wtcls_sigmoid, self).__init__()
        # self.is_batchnorm = is_batchnorm 
        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # upsampling
        self.up_concat4 = UnetUp3_CT_sigmoid(self.ft_chns[4], self.ft_chns[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT_sigmoid(self.ft_chns[3], self.ft_chns[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT_sigmoid(self.ft_chns[2], self.ft_chns[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT_sigmoid(self.ft_chns[1], self.ft_chns[0], is_batchnorm)

        # final conv (without any concat)
        self.dropout = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
                
    def forward(self, features, consist=False):
        conv1, conv2, conv3, conv4, center = features[:]
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout(up1)
        if consist:
            return center, up1
        return up1

class unet_3D_sigmoid(nn.Module):
    def __init__(self, in_chns, class_num):
        super(unet_3D_sigmoid, self).__init__()
        params = {'in_chns': in_chns,
                  'is_batchnorm': True,
                  'feature_scale': 4,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.params = params
        self.encoder = Encoder_sigmoid(self.params)
        self.decoder = Decoder_wtcls_sigmoid(self.params)
        self.classifier = nn.Conv3d(self.params['feature_chns'][0], class_num, 1)

    def forward(self, inputs, comp_drop=None, feature_need=False, consist=False):
        features = self.encoder(inputs)
        if comp_drop != None:
            for i in range(0, len(features)):
                features[i] = features[i] * comp_drop[i].unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            out = self.decoder(features)
            return out
        feature_final = self.decoder(features, consist=consist)
        if consist:
            center, feature_final = feature_final[0], feature_final[1]
        out = self.classifier(feature_final)
        if feature_need:
            if consist:
                return out, center, feature_final
            return out, feature_final
        return out

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        
        self.conv1 = UnetConv3(self.in_chns, self.ft_chns[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(self.ft_chns[0], self.ft_chns[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(self.ft_chns[1], self.ft_chns[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(self.ft_chns[2], self.ft_chns[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(self.ft_chns[3], self.ft_chns[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        
        center = self.center(maxpool4)
        center = self.dropout(center)
        
        return [conv1, conv2, conv3, conv4, center]

class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2, deep=False):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        self.deep = torch.cat([outputs1, outputs2], 1)
        return self.conv(self.deep)
    
class Decoder_wtcls(nn.Module):
    def __init__(self, params, in_channels=3, is_batchnorm=True):
        super(Decoder_wtcls, self).__init__()
        # self.is_batchnorm = is_batchnorm 
        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # upsampling
        self.up_concat4 = UnetUp3_CT(self.ft_chns[4], self.ft_chns[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(self.ft_chns[3], self.ft_chns[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(self.ft_chns[2], self.ft_chns[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(self.ft_chns[1], self.ft_chns[0], is_batchnorm)

        # final conv (without any concat)
        self.dropout = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
                
    def forward(self, features, deep=False, consist=False):
        conv1, conv2, conv3, conv4, center = features[:]
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2, deep=deep)
        self.center = center
        return up1
    
class unet_3D_wtcls(nn.Module):
    def __init__(self, in_chns, class_num):
        super(unet_3D_wtcls, self).__init__()
        params = {'in_chns': in_chns,
                  'is_batchnorm': True,
                  'feature_scale': 4,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.params = params
        self.encoder = Encoder(self.params)
        self.decoder = Decoder_wtcls(self.params)
        self.classifier = nn.Conv3d(self.params['feature_chns'][0], class_num, 1)

    def forward(self, inputs, feature_need=False):
        features = self.encoder(inputs)
        feature_final = self.decoder(features)

        out = self.classifier(feature_final)
        
        if feature_need:
            return out, feature_final

        return out
    
class Projector(nn.Module):
    def __init__(self, in_chns, hidden_chns, out_num):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Conv3d(in_chns, hidden_chns, kernel_size=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_chns), 
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_chns, out_num, kernel_size=1))
    
    def forward(self, inputs):
        projects = self.projector(inputs)
        
        return projects

class Decoder(nn.Module):
    def __init__(self, params, in_channels=3, is_batchnorm=True):
        super(Decoder, self).__init__()
        # self.is_batchnorm = is_batchnorm 
        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # upsampling
        self.up_concat4 = UnetUp3_CT(self.ft_chns[4], self.ft_chns[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(self.ft_chns[3], self.ft_chns[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(self.ft_chns[2], self.ft_chns[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(self.ft_chns[1], self.ft_chns[0], is_batchnorm)

        # final conv (without any concat)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Conv3d(self.params['feature_chns'][0], self.n_class, 1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
                
    def forward(self, features, deep=False, consist=False):
        conv1, conv2, conv3, conv4, center = features[:]
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2, deep=deep)
        self.center = center
        out = self.classifier(up1)
        return out
    
class unet_3D_mt(nn.Module):
    def __init__(self, in_chns, class_num):
        super(unet_3D_mt, self).__init__()
        params = {'in_chns': in_chns,
                  'is_batchnorm': True,
                  'feature_scale': 4,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def forward(self, inputs, comp_drop=None, up1_s=False):
        features = self.encoder(inputs)
        if comp_drop:
            bs = features[0].shape[0]
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            for i in range(0, len(features)):   
                dim = features[i].shape[1]
                dropout_mask1 = self.binomial.sample((bs, dim)).cuda() * 2.0
                dropout_mask2 = 2.0 - dropout_mask1
                dropout_mask1[kept_indexes, :] = 1.0
                dropout_mask2[kept_indexes, :] = 1.0

                # # unimatch
                # dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
                # features[i] = features[i] * dropout_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                features[i] = features[i] * dropout_mask1.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            out, up1 = self.decoder(features)
            return out
        
        out = self.decoder(features)
        
        if not up1_s:
            return out
        
        return out, up1

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p