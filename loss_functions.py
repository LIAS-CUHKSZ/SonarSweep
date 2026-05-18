from __future__ import division
import torch

def compute_errors_test(gt, pred):
    '''
    Compute test errors over valid pixels.

    Args:
        gt: Ground-truth depth values.
        pred: Predicted depth values.

    Returns:
        abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3.
    '''
    
    abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)
    
    for current_gt, current_pred in zip(gt, pred):
        # Build the valid-region mask.
        valid_mask = (current_gt >= 0.5) & (current_gt <= 4.5) & (current_gt == current_gt)
        valid_mask[:, :20] = False
        valid_mask[:, -20:] = False
        valid_gt = current_gt[valid_mask]
        valid_pred = current_pred[valid_mask]
        
        
        # Threshold accuracy.
        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()
        
        # Error metrics.
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)
        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)
        
        # RMSE.
        rmse_val = (valid_gt - valid_pred) ** 2
        rmse += torch.sqrt(torch.mean(rmse_val))
        
        # Log RMSE.
        rmse_log_val = (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
        rmse_log += torch.sqrt(torch.mean(rmse_log_val))
    
    return [float(metric / batch_size) for metric in [abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3]]

def compute_errors_train(gt, pred, valid):
    '''
        # Threshold accuracy
        thresh = max(gt/pred, pred/gt) < threshold  closer to 1 is better, less to 1 too much is worse 

        # Absolute error
        abs_diff = |gt - pred|

        # Absolute relative error
        abs_rel = |gt - pred| / gt

        # Squared relative error
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
