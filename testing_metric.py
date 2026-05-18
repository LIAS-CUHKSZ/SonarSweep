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
        valid_mask[:4, :] = False
        valid_mask[-4:, :] = False
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

import numpy as np
from collections import defaultdict

class DepthEvaluator:
    """
    Accumulates depth-estimation metrics across batches and depth bins.
    """
    def __init__(self, bins=None, min_depth=0.5, max_depth=4.5, base_dir=None):
        """
        Args:
            bins (list of tuples, optional): Custom depth-bin ranges. If None,
                default bins are used.
            min_depth (float): Minimum evaluated depth.
            max_depth (float): Maximum evaluated depth.
            base_dir (str, optional): Directory used when saving results.
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.base_dir = base_dir if base_dir is not None else "."
        
        # Default depth bins.
        if bins is None:
            self.bins = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5)]
        else:
            self.bins = bins
            
        self.metric_names = [
            'abs_rel', 'abs_diff', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'
        ]
        
        self.reset()

    def reset(self):
        """Reset all accumulators before a new evaluation."""
        # Overall metric accumulators.
        self.overall_metrics = np.zeros(len(self.metric_names), dtype=np.float64)
        self.num_samples = 0
        
        # Binned metric accumulators.
        self.binned_metrics = defaultdict(lambda: np.zeros(len(self.metric_names), dtype=np.float64))
        self.binned_counts = defaultdict(int)
        
        # Pixel counters used to compute bin percentages.
        self.total_valid_pixels = 0
        self.binned_pixel_counts = defaultdict(int)

    def _compute_metrics(self, gt, pred):
        """
        Compute metrics for a set of valid pixels.

        Args:
            gt (torch.Tensor): 1D Tensor of valid ground truth depths.
            pred (torch.Tensor): 1D Tensor of valid predicted depths.

        Returns:
            numpy.ndarray: Metric values.
        """
        if gt.numel() == 0:
            return np.zeros(len(self.metric_names), dtype=np.float64)

        # Threshold accuracy.
        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        # Error metrics.
        abs_diff = torch.mean(torch.abs(gt - pred))
        abs_rel = torch.mean(torch.abs(gt - pred) / gt)
        sq_rel = torch.mean(((gt - pred) ** 2) / gt)

        # RMSE
        rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
        
        # RMSE log
        # Add a small epsilon to avoid log(0).
        rmse_log = torch.sqrt(torch.mean((torch.log(gt + 1e-8) - torch.log(pred + 1e-8)) ** 2))

        metrics = np.array([
            abs_rel.item(), abs_diff.item(), sq_rel.item(),
            rmse.item(), rmse_log.item(),
            a1.item(), a2.item(), a3.item()
        ])
        
        return metrics

    def update(self, gt, pred):
        """
        Update the accumulated metrics with a new batch.

        Args:
            gt (torch.Tensor): Ground-truth depth maps, shaped (B, 1, H, W) or (B, H, W).
            pred (torch.Tensor): Predicted depth maps, shaped (B, 1, H, W) or (B, H, W).
        """
        batch_size = gt.size(0)
        
        for i in range(batch_size):
            current_gt = gt[i].squeeze()
            current_pred = pred[i].squeeze()

            # 1. Overall evaluation.
            # Build the valid-pixel mask.
            valid_mask = (current_gt >= self.min_depth) & (current_gt <= self.max_depth)
            
            # Keep the original border filtering used during evaluation.
            valid_mask[:4, :] = False
            valid_mask[-4:, :] = False
            valid_mask[:, :20] = False
            valid_mask[:, -20:] = False

            valid_gt = current_gt[valid_mask]
            valid_pred = current_pred[valid_mask]
            
            # Skip samples without valid pixels.
            if valid_gt.numel() == 0:
                continue

            # Compute and accumulate overall metrics.
            overall_metrics_sample = self._compute_metrics(valid_gt, valid_pred)
            self.overall_metrics += overall_metrics_sample
            self.num_samples += 1
            
            # Count total valid pixels.
            self.total_valid_pixels += valid_gt.numel()
            
            # 2. Binned evaluation.
            for bin_range in self.bins:
                min_d, max_d = bin_range
                
                # Build the mask for the current depth bin.
                bin_mask = (valid_gt >= min_d) & (valid_gt < max_d)
                
                gt_bin = valid_gt[bin_mask]
                pred_bin = valid_pred[bin_mask]
                
                if gt_bin.numel() > 500: # only consider bins with more than 1000 valid pixels, to ensure statistical significance
                    binned_metrics_sample = self._compute_metrics(gt_bin, pred_bin)
                    self.binned_metrics[bin_range] += binned_metrics_sample
                    self.binned_counts[bin_range] += 1
                    
                    # Count pixels in each bin.
                    self.binned_pixel_counts[bin_range] += gt_bin.numel()
    
    def get_results(self):
        """
        Compute and return the final averaged metrics.

        Returns:
            dict: A dictionary containing overall and binned results.
        """
        if self.num_samples == 0:
            print("Warning: No samples were processed.")
            return None

        # Overall averages.
        avg_overall_metrics = self.overall_metrics / self.num_samples
        overall_results = dict(zip(self.metric_names, avg_overall_metrics))
        
        # Binned averages and percentages.
        binned_results = {}
        for bin_range, total_metrics in self.binned_metrics.items():
            count = self.binned_counts[bin_range]
            if count > 0:
                avg_binned_metrics = total_metrics / count
                pixel_count = self.binned_pixel_counts[bin_range]
                percentage = (pixel_count / self.total_valid_pixels) * 100 if self.total_valid_pixels > 0 else 0
                
                bin_results = dict(zip(self.metric_names, avg_binned_metrics))
                bin_results['percentage'] = percentage
                bin_results['pixel_count'] = pixel_count
                binned_results[str(bin_range)] = bin_results
        
        return {
            'overall': overall_results,
            'binned': binned_results,
            'total_pixels': self.total_valid_pixels
        }

    def save_results(self, filename="evaluation_results.txt"):
        """Save evaluation results to a text file.
        
        Args:
            filename (str): Output filename.
        """
        import os
        
        results = self.get_results()
        if results is None:
            return

        # Ensure the output directory exists.
        os.makedirs(self.base_dir, exist_ok=True)
        filepath = os.path.join(self.base_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            header = " | ".join([f"{name:^10}" for name in self.metric_names])
            header += " | " + f"{'Percentage':^12} | {'PixelCount':^12}"
            separator = "-" * len(header)
            
            f.write(separator + "\n")
            f.write(f"{'Category':<15} | {header}\n")
            f.write(separator + "\n")

            # Overall results.
            overall = results['overall']
            metrics_str = " | ".join([f"{overall[name]:^10.4f}" for name in self.metric_names])
            f.write(f"{'Overall':<15} | {metrics_str} | {'100.00%':^12} | {results['total_pixels']:^12}\n")
            
            # Binned results.
            binned = results['binned']
            for bin_range_str, bin_metrics in sorted(binned.items()):
                metrics_str = " | ".join([f"{bin_metrics[name]:^10.4f}" for name in self.metric_names])
                percentage_str = f"{bin_metrics['percentage']:^10.2f}%"
                pixel_count_str = f"{bin_metrics['pixel_count']:^12}"
                f.write(f"{bin_range_str:<15} | {metrics_str} | {percentage_str:^12} | {pixel_count_str}\n")
            
            f.write(separator + "\n")
        
        print(f"Results saved to: {filepath}")

    def display_results(self):
        """Print formatted evaluation results."""
        results = self.get_results()
        if results is None:
            return

        header = " | ".join([f"{name:^10}" for name in self.metric_names])
        header += " | " + f"{'Percentage':^12} | {'PixelCount':^12}"
        print("-" * len(header))
        print(f"{'Category':<15} | {header}")
        print("-" * len(header))

        # Overall results.
        overall = results['overall']
        metrics_str = " | ".join([f"{overall[name]:^10.4f}" for name in self.metric_names])
        print(f"{'Overall':<15} | {metrics_str} | {'100.00%':^12} | {results['total_pixels']:^12}")
        
        # Binned results.
        binned = results['binned']
        for bin_range_str, bin_metrics in sorted(binned.items()):
            metrics_str = " | ".join([f"{bin_metrics[name]:^10.4f}" for name in self.metric_names])
            percentage_str = f"{bin_metrics['percentage']:^10.2f}%"
            pixel_count_str = f"{bin_metrics['pixel_count']:^12}"
            print(f"{bin_range_str:<15} | {metrics_str} | {percentage_str:^12} | {pixel_count_str}")
        
        print("-" * len(header))
