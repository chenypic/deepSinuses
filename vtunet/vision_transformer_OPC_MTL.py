# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
from torch.nn import init, Parameter
from .vt_unet_MTL import SwinTransformerSys3D

logger = logging.getLogger(__name__)



class Surv_network(nn.Module):
    def __init__(self):
        super(Surv_network, self).__init__()
        self.avg_pool_3d = nn.AvgPool3d((4,4,4), 1)
        self.max_pool_3d = nn.MaxPool3d((4,4,4), 1)
        self.Hidder_layer_1 = nn.Linear(1536, 256)
        self.relu1 = nn.ReLU(True)
        self.Hidder_layer_2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU(True)
        self.drop_layer = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(64, 1)
        #self.softmax = nn.Softmax(dim=1)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        self.act = nn.Sigmoid()

    def forward(self,x4_1):
        x = self.feature_fusion_layer(x4_1)
        x = self.drop_layer(x)
        x = self.Hidder_layer_1(x)
        x = self.relu1(x)
        x = self.Hidder_layer_2(x)
        x = self.relu2(x)
        hazard = self.classifier(x)
        
        hazard = self.act(hazard)

        # hazard = hazard * self.output_range + self.output_shift

        return x, hazard

    def feature_fusion_layer(self,x4_1):
        x4_1_avg = self.avg_pool_3d(x4_1)
        x4_1_max = self.max_pool_3d(x4_1)

        x4_1_avg = x4_1_avg.view(x4_1_avg.size(0), -1)
        x4_1_max = x4_1_max.view(x4_1_max.size(0), -1)

        return torch.cat([x4_1_avg,x4_1_max], dim=1)



class VTUNet(nn.Module):
    def __init__(self, config, num_classes=1, zero_head=False, embed_dim=96, win_size=7): # 1类，分割1类
        super(VTUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.win_size = (self.win_size,self.win_size,self.win_size)

        self.swin_unet = SwinTransformerSys3D(img_size=(128, 128, 128),
                                            patch_size=(4, 4, 4),
                                            in_chans=1, 
                                            num_classes=self.num_classes,
                                            embed_dim=self.embed_dim,
                                            depths=[2, 2, 2, 1],
                                            depths_decoder=[1, 2, 2, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=self.win_size,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.,
                                            attn_drop_rate=0.,
                                            drop_path_rate=0.1,
                                            norm_layer=nn.LayerNorm,
                                            patch_norm=True,
                                            use_checkpoint=False,
                                            frozen_stages=-1,
                                            final_upsample="expand_first") # in_chans由4改成1；

    def forward(self, x):
        logits,bottle_features = self.swin_unet(x)
        return logits,bottle_features

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin_unet.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")




# class Surv_network_qian(nn.Module):
#     def __init__(self):
#         super(Surv_network_qian, self).__init__()
#         self.avg_pool_3d = nn.AvgPool3d((8,8,8), 1)
#         self.max_pool_3d = nn.MaxPool3d((8,8,8), 1)
#         self.Hidder_layer_1 = nn.Linear(768, 256)
#         self.relu1 = nn.ReLU(True)
#         self.Hidder_layer_2 = nn.Linear(256, 64)
#         self.relu2 = nn.ReLU(True)
#         self.drop_layer = nn.Dropout(p=0.2)
#         self.classifier = nn.Linear(64, 1)
#         #self.softmax = nn.Softmax(dim=1)

#         self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
#         self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

#         self.act = nn.Sigmoid()

#     def forward(self,x4_1):
#         x = self.feature_fusion_layer(x4_1)
#         x = self.drop_layer(x)
#         x = self.Hidder_layer_1(x)
#         x = self.relu1(x)
#         x = self.Hidder_layer_2(x)
#         x = self.relu2(x)
#         hazard = self.classifier(x)
        
#         hazard = self.act(hazard)

#         # hazard = hazard * self.output_range + self.output_shift

#         return x, hazard

#     def feature_fusion_layer(self,x4_1):
#         x4_1_avg = self.avg_pool_3d(x4_1)
#         x4_1_max = self.max_pool_3d(x4_1)

#         x4_1_avg = x4_1_avg.view(x4_1_avg.size(0), -1)
#         x4_1_max = x4_1_max.view(x4_1_max.size(0), -1)

#         return torch.cat([x4_1_avg,x4_1_max], dim=1)