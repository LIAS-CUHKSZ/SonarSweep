from __future__ import division
import torch
from torch import nn
import numpy as np

def compute_errors_test(gt, pred):
    '''
    计算测试误差，输入应该是已经过滤的有效像素
    参数:
        gt: 真实深度值 (可以是1D张量的有效像素，或2D/3D张量的批次数据)
        pred: 预测深度值
    返回:
        abs_rel: 相对绝对误差
        abs_diff: 绝对误差
        sq_rel: 相对平方误差
        rmse: 均方根误差
        rmse_log: 对数空间的均方根误差
        a1, a2, a3: 阈值准确度
    '''
    
    abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)
    
    for current_gt, current_pred in zip(gt, pred):
        # 创建有效区域掩码
        valid_mask = (current_gt >= 0.5) & (current_gt <= 4.5) & (current_gt == current_gt)
        valid_mask[:, :20] = False  # 对est mask也应用相同过滤
        valid_mask[:, -20:] = False
        valid_gt = current_gt[valid_mask]
        valid_pred = current_pred[valid_mask]
        
        
        # 阈值准确度计算
        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()
        
        # 误差计算
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)
        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)
        
        # RMSE计算
        rmse_val = (valid_gt - valid_pred) ** 2
        rmse += torch.sqrt(torch.mean(rmse_val))
        
        # RMSE log计算
        rmse_log_val = (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
        rmse_log += torch.sqrt(torch.mean(rmse_log_val))
    
    return [float(metric / batch_size) for metric in [abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3]]

def compute_errors_train(gt, pred, valid):
    '''
        # 阈值准确度
        thresh = max(gt/pred, pred/gt) < threshold  closer to 1 is better, less to 1 too much is worse 

        # 绝对误差
        abs_diff = |gt - pred|

        # 相对绝对误差
        abs_rel = |gt - pred| / gt

        # 相对平方误差
        sq_rel = (gt - pred)² / gt
    '''

    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    for current_gt, current_pred, current_valid in zip(gt, pred, valid):
        valid_gt = current_gt[current_valid]
        valid_pred = current_pred[current_valid]

        if len(valid_gt) == 0:
            continue
        else:
            thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
            a1 += (thresh < 1.1).float().mean()
            a2 += (thresh < 1.1 ** 2).float().mean()
            a3 += (thresh < 1.1 ** 3).float().mean()

            abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
            abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

            sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [float(metric / batch_size)for metric in [abs_rel, abs_diff, sq_rel, a1, a2, a3]]
