# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 trilinear=True):
        super(UpBlock, self).__init__()
        self.trilinear = trilinear
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class UNetBlock(nn.Module):
    def __init__(self,in_channels, out_channels, acti_func, acti_func_param):
        super(UNetBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 3, 
                padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3, 
                padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.in_chns   = 1
        self.ft_chns   = [16, 32, 64, 128, 256]
        self.n_class   = 1
        self.trilinear = True
        self.dropout   = [False,False,False,False,False]
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        if(len(self.ft_chns) == 5):
          self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
          self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 
               dropout_p = 0.0, trilinear=self.trilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 
               dropout_p = 0.0, trilinear=self.trilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 
               dropout_p = 0.0, trilinear=self.trilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 
               dropout_p = 0.0, trilinear=self.trilinear) 
    
        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class,  
            kernel_size = 1, padding = 0)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        if(len(self.ft_chns) == 5):
          x4_bottle = self.down4(x3)
          x = self.up1(x4_bottle, x3)
        else:
          x = x3
          x4_bottle = x3

        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)

        return output, x4_bottle




class Surv_network_qian_unet(nn.Module):
    def __init__(self):
        super(Surv_network_qian_unet, self).__init__()
        self.avg_pool_3d = nn.AvgPool3d((8,8,8), 1)
        self.max_pool_3d = nn.MaxPool3d((8,8,8), 1)
        self.Hidder_layer_1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU(True)
        self.Hidder_layer_2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU(True)
        self.drop_layer = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(64, 2)
        #self.softmax = nn.Softmax(dim=1)

        #self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        #self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        # self.act = nn.Sigmoid()

    def forward(self,x4_1):
        x = self.feature_fusion_layer(x4_1)
        x = self.drop_layer(x)
        x = self.Hidder_layer_1(x)
        x = self.relu1(x)
        x = self.Hidder_layer_2(x)
        x = self.relu2(x)
        x = self.classifier(x)
        
        # hazard = self.act(hazard)

        # hazard = hazard * self.output_range + self.output_shift

        return x


    def feature_fusion_layer(self,x4_1):
        x4_1_avg = self.avg_pool_3d(x4_1)
        x4_1_max = self.max_pool_3d(x4_1)

        x4_1_avg = x4_1_avg.view(x4_1_avg.size(0), -1)
        x4_1_max = x4_1_max.view(x4_1_max.size(0), -1)

        return torch.cat([x4_1_avg,x4_1_max], dim=1)









if __name__ == "__main__":
    params = {'in_chns':4,
              'class_num': 2,
              'feature_chns':[2, 8, 32, 64],
              'dropout' : [0, 0, 0, 0.5],
              'trilinear': True}
    Net = UNet3D(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    y = y.detach().numpy()
    print(y.shape)
