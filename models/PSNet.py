from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .submodule import *

from .inverse_warp import warp_camera_to_sonar
from .recover_depth import compute_depth_norm_from_plane
import numpy as np

def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

class PSNet(nn.Module):
    def __init__(self, nlabel=48, mindepth=0.5, label_factor=1.05, alpha=np.deg2rad(45)):
        super(PSNet, self).__init__()
        self.nlabel = nlabel
 
        # Register mindepth and label_factor as buffers
        self.register_buffer('mindepth', torch.tensor(mindepth, dtype=torch.float32))
        self.register_buffer('label_factor', torch.tensor(label_factor, dtype=torch.float32))
            
        
        # Register the candidate plane depths.
        self.register_buffer('depths', 
            torch.tensor([mindepth * label_factor**i for i in range(nlabel)], 
                        dtype=torch.float32))
        
        # Register the plane angle.
        self.register_buffer('alpha',
            torch.tensor(alpha, dtype=torch.float32))
        
        # Register the plane normal.
        self.register_buffer('n_s',
            torch.tensor([0, torch.sin(self.alpha), torch.cos(self.alpha)], 
                        dtype=torch.float32).unsqueeze(-1))
        
        self.rgb_feature_extraction = feature_extraction()
        self.sonar_feature_extraction = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1), # dilation
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def initialize_feature_extractors(self):
        """
        Initialize the camera and sonar feature extraction networks.
        """
        print("=> Re-initializing rgb_feature_extraction and sonar_feature_extraction")
        for m in self.rgb_feature_extraction.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for LeakyReLU layers.
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.sonar_feature_extraction.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, cam_img_var, sonar_rect_img_var, K_var, T_sonar_from_cam_var, distance_range_var, theta_range_var):  # reference + j * targets
        
        # We downsample sonar feature by 2 so we need to change K accordingly (please refer to the model/submodule.py)
        scale_factor = 2
        K_cam_scaled = K_var.clone()
        K_cam_scaled[:, :2, :] /= scale_factor
        
        rgb_img_feature = self.rgb_feature_extraction(cam_img_var)  # torch.Size([1, 3, 480, 640])  =>  torch.Size([1, 32, 120, 160])
        sonar_feature   = self.sonar_feature_extraction(sonar_rect_img_var)

        # Cost shape
        # Batch   feature_channel   label/layers   width   height
        cost = torch.zeros((rgb_img_feature.size()[0], rgb_img_feature.size()[1]*2, self.nlabel,  rgb_img_feature.size()[2],  rgb_img_feature.size()[3])).cuda()
        
        for i, depth in enumerate(self.depths):
            
            # warp the sonar feature to camera view 
            warped_sonar_feature = warp_camera_to_sonar(sonar_feature, K_cam_scaled, T_sonar_from_cam_var, rgb_img_feature.size(), 
                                                            depth, distance_range_var, theta_range_var, self.alpha)
            # torch.Size([1, 1, 68, 366]) => torch.Size([1, 32, 34, 184])
            
            cost[:, :warped_sonar_feature.size()[1], i, :,:] = rgb_img_feature
            cost[:, warped_sonar_feature.size()[1]:, i, :,:] = warped_sonar_feature

            
        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0 
        cost0 = self.dres3(cost0) + cost0 
        cost0 = self.dres4(cost0) + cost0
        costs = self.classify(cost0)  # half

        
        costss = torch.zeros((rgb_img_feature.size()[0], 1, self.nlabel,  rgb_img_feature.size()[2],  rgb_img_feature.size()[3])).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([rgb_img_feature, costt],1)) + costt

        ##################################################
        costs = F.upsample(costs, [self.nlabel,cam_img_var.size()[2],cam_img_var.size()[3]], mode='trilinear')
        costs = torch.squeeze(costs,1)
        pred0_probability = F.softmax(costs, dim=1, dtype=torch.float32)  # convert cost volume to probability distribution
        pred0, variance0 = DisparityRegression(self.nlabel)(pred0_probability) # give depth plane index
        
        costss = F.upsample(costss, [self.nlabel,cam_img_var.size()[2],cam_img_var.size()[3]], mode='trilinear')
        costss = torch.squeeze(costss,1)
        pred_probability = F.softmax(costss, dim=1, dtype=torch.float32)
        pred, variance = DisparityRegression(self.nlabel)(pred_probability) # torch.Size([B, 64, 480, 640])
        
        depth0_pseudo_plane = self.mindepth * self.label_factor**(pred0.unsqueeze(1))          # need to recover depth from plane index
        depth0 = compute_depth_norm_from_plane(depth0_pseudo_plane, K_var, T_sonar_from_cam_var, self.n_s, self.alpha)
        
        depth_pseudo_plane = self.mindepth * self.label_factor**(pred.unsqueeze(1))
        depth = compute_depth_norm_from_plane(depth_pseudo_plane, K_var, T_sonar_from_cam_var, self.n_s, self.alpha)
        ##################################################

        if self.training:
            return depth0, depth
        else:
            return depth, variance
