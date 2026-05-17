from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(1, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 1,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)


        self.branch1 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4,4)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2,2)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):                 # torch.Size([1, 3,  256, 512])
        output      = self.firstconv(x)   # torch.Size([1, 32, 128, 256])  (H/2, W/2) 下采样
        output      = self.layer1(output)  # torch.Size([1, 32, 128, 256]) 
        output_raw  = self.layer2(output)  # torch.Size([1, 64, 64, 128]) (H/4, W/4)  下采样
        output      = self.layer3(output_raw) # torch.Size([1, 128, 64, 128])
        output_skip = self.layer4(output)     # torch.Size([1, 128, 64, 128])


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


    
class DisparityRegression(nn.Module):
    def __init__(self, maxdisp, device='cuda'):
        super(DisparityRegression, self).__init__()
        # 创建视差值张量 d: [0, 1, 2, ...]
        disp_values = torch.arange(maxdisp, dtype=torch.float32, device=device).view(1, maxdisp, 1, 1)
        self.register_buffer('disp', disp_values) # 使用 register_buffer 注册，它不会被视为模型参数

        # 创建视差值平方的张量 d^2: [0, 1, 4, ...]
        disp_sq_values = disp_values ** 2
        self.register_buffer('disp_sq', disp_sq_values)

    def forward(self, x):
        # x 是 softmax 的输出，即概率体 P(d)
        # x 的形状: [B, D, H, W]
        
        # 计算期望 E[d] = Σ [d * P(d)]
        # self.disp 的形状是 [1, D, 1, 1]，会自动广播以匹配 x
        disparity = torch.sum(x * self.disp, 1)

        # 计算 E[d^2] = Σ [d^2 * P(d)]
        # self.disp_sq 的形状是 [1, D, 1, 1]，也会自动广播
        disparity_sq_expectation = torch.sum(x * self.disp_sq, 1)
        
        # 计算方差 Var(d) = E[d^2] - (E[d])^2
        variance = disparity_sq_expectation - disparity.pow(2)

        # 返回视差图（期望）和不确定性图（方差）
        # 两者的形状都是 [B, H, W]
        return disparity, variance



# ==============================================================================
# 以下是用于分析网络架构的 main 函数
# ==============================================================================
def analyze_network_architecture():
    """
    此函数实例化 feature_extraction 和 soanr_feature_extraction 网络，
    并详细分析它们的架构、数据流、维度变化和参数量。
    """
    print("="*80)
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("="*80)

    # --------------------------------------------------------------------------
    # 1. 分析 'feature_extraction' 网络 (用于1通道图像)
    # --------------------------------------------------------------------------
    print("\n--- 1. 分析 'feature_extraction' 网络 (设计用于1通道图像) ---\n")
    
    # 实例化模型
    feature_extraction_model = feature_extraction()

    # 计算参数量
    total_params = sum(p.numel() for p in feature_extraction_model.parameters())
    trainable_params = sum(p.numel() for p in feature_extraction_model.parameters() if p.requires_grad)
    
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小估计: {total_params * 4 / 1024 / 1024:.2f} MB (假设每个参数4字节)")
    
    # 各子模块参数贡献分析
    print("\n各子模块参数贡献:")
    print(f"{'模块名':<20} {'参数量':>15} {'占比 (%)':>15}")
    print("-" * 55)
    
    for name, module in feature_extraction_model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        if total_params > 0:
            proportion = (module_params / total_params) * 100
            print(f"{name:<20} {module_params:>15,} {proportion:>14.2f}%")
    print("-" * 55)

    # 测试前向传播
    dummy_input_img = torch.randn(1, 1, 68, 368)
    print(f"\n输入张量尺寸: {dummy_input_img.shape}")
    
    try:
        final_output = feature_extraction_model(dummy_input_img)
        print(f"成功执行前向传播！")
        print(f"最终输出特征图尺寸: {final_output.shape}")
    except Exception as e:
        print(f"执行前向传播时出错: {e}")

    dummy_input_img = torch.randn(1, 1, 112, 640)
    print(f"\n输入张量尺寸: {dummy_input_img.shape}")
    
    try:
        final_output = feature_extraction_model(dummy_input_img)
        print(f"成功执行前向传播！")
        print(f"最终输出特征图尺寸: {final_output.shape}")
    except Exception as e:
        print(f"执行前向传播时出错: {e}")
        
if __name__ == "__main__":
    analyze_network_architecture()


# --- 1. 分析 'feature_extraction' 网络 (设计用于1通道图像) ---

# 模型总参数量: 3,338,976
# 可训练参数量: 3,338,976
# 模型大小估计: 12.74 MB (假设每个参数4字节)

# 各子模块参数贡献:
# 模块名                              参数量          占比 (%)
# -------------------------------------------------------
# firstconv                     18,912           0.57%
# layer1                        55,680           1.67%
# layer2                     1,167,488          34.97%
# layer3                       820,992          24.59%
# layer4                       886,272          26.54%
# branch1                        4,160           0.12%
# branch2                        4,160           0.12%
# branch3                        4,160           0.12%
# branch4                        4,160           0.12%
# lastconv                     372,992          11.17%
# -------------------------------------------------------

# 输入张量尺寸: torch.Size([1, 1, 68, 368])
# /home/clp/.local/lib/python3.12/site-packages/torch/nn/functional.py:3809: UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
#   warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
# 成功执行前向传播！
# 最终输出特征图尺寸: torch.Size([1, 32, 34, 184])

# 输入张量尺寸: torch.Size([1, 1, 112, 640])
# 成功执行前向传播！
# 最终输出特征图尺寸: torch.Size([1, 32, 56, 320])