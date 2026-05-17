import torch

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
        valid_mask[:4, :] = False  # 过滤上边缘4像素
        valid_mask[-4:, :] = False  # 过滤下边缘4像素
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

# 能否帮我写一个升级版的函数，可以计算0.5-1.5 1.5-2.5 2.5-3.5 3.5-4.5的深度误差


import torch
import numpy as np
from collections import defaultdict

class DepthEvaluator:
    """
    一个用于深度估计模型评估的类。
    它可以在多个批次上累积指标，并提供按深度范围分桶的评估。
    """
    def __init__(self, bins=None, min_depth=0.5, max_depth=4.5, base_dir=None):
        """
        初始化评估器。

        参数:
            bins (list of tuples, optional): 自定义深度分桶范围。
                                            例如: [(0.5, 1.5), (1.5, 2.5), (2.5, 4.5)]
                                            如果为 None, 将使用默认分桶。
            min_depth (float): 评估的最小深度。
            max_depth (float): 评估的最大深度。
            base_dir (str, optional): 保存结果的基本目录路径。如果为 None，将保存到当前目录。
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.base_dir = base_dir if base_dir is not None else "."
        
        # 定义默认的深度分桶范围
        if bins is None:
            self.bins = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5)]
        else:
            self.bins = bins
            
        self.metric_names = [
            'abs_rel', 'abs_diff', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'
        ]
        
        self.reset()

    def reset(self):
        """重置所有累加器，以便开始新的评估。"""
        # 整体指标累加器
        self.overall_metrics = np.zeros(len(self.metric_names), dtype=np.float64)
        self.num_samples = 0
        
        # 分桶指标累加器
        # 使用 defaultdict 简化代码，如果一个桶没有数据，它的值将是 0 数组
        self.binned_metrics = defaultdict(lambda: np.zeros(len(self.metric_names), dtype=np.float64))
        self.binned_counts = defaultdict(int)
        
        # 像素计数累加器，用于计算百分比
        self.total_valid_pixels = 0
        self.binned_pixel_counts = defaultdict(int)

    def _compute_metrics(self, gt, pred):
        """
        为一组给定的有效像素计算指标。
        参数:
            gt (torch.Tensor): 1D Tensor of valid ground truth depths.
            pred (torch.Tensor): 1D Tensor of valid predicted depths.
        返回:
            numpy.ndarray: 包含所有指标计算结果的数组。
        """
        if gt.numel() == 0:
            return np.zeros(len(self.metric_names), dtype=np.float64)

        # 阈值精度
        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        # 误差
        abs_diff = torch.mean(torch.abs(gt - pred))
        abs_rel = torch.mean(torch.abs(gt - pred) / gt)
        sq_rel = torch.mean(((gt - pred) ** 2) / gt)

        # RMSE
        rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
        
        # RMSE log
        # 添加一个小的 epsilon 防止 log(0)
        rmse_log = torch.sqrt(torch.mean((torch.log(gt + 1e-8) - torch.log(pred + 1e-8)) ** 2))

        metrics = np.array([
            abs_rel.item(), abs_diff.item(), sq_rel.item(),
            rmse.item(), rmse_log.item(),
            a1.item(), a2.item(), a3.item()
        ])
        
        return metrics

    def update(self, gt, pred):
        """
        使用一个新的批次数据来更新累积的指标。
        参数:
            gt (torch.Tensor): 真实深度图 (B, 1, H, W) or (B, H, W)
            pred (torch.Tensor): 预测深度图 (B, 1, H, W) or (B, H, W)
        """
        batch_size = gt.size(0)
        
        for i in range(batch_size):
            current_gt = gt[i].squeeze()
            current_pred = pred[i].squeeze()

            # 1. 整体评估 (Overall Evaluation)
            # 创建有效像素掩码
            valid_mask = (current_gt >= self.min_depth) & (current_gt <= self.max_depth)
            
            # 你的代码中包含了对图像边缘的过滤，这里保留
            valid_mask[:4, :] = False
            valid_mask[-4:, :] = False
            valid_mask[:, :20] = False
            valid_mask[:, -20:] = False

            valid_gt = current_gt[valid_mask]
            valid_pred = current_pred[valid_mask]
            
            # 如果没有有效像素，则跳过这个样本
            if valid_gt.numel() == 0:
                continue

            # 计算并累积整体指标
            overall_metrics_sample = self._compute_metrics(valid_gt, valid_pred)
            self.overall_metrics += overall_metrics_sample
            self.num_samples += 1
            
            # 统计总的有效像素数
            self.total_valid_pixels += valid_gt.numel()
            
            # 2. 分桶评估 (Binned Evaluation)
            for bin_range in self.bins:
                min_d, max_d = bin_range
                
                # 创建当前桶的掩码
                bin_mask = (valid_gt >= min_d) & (valid_gt < max_d)
                
                gt_bin = valid_gt[bin_mask]
                pred_bin = valid_pred[bin_mask]
                
                if gt_bin.numel() > 500: # only consider bins with more than 1000 valid pixels, to ensure statistical significance
                    binned_metrics_sample = self._compute_metrics(gt_bin, pred_bin)
                    self.binned_metrics[bin_range] += binned_metrics_sample
                    self.binned_counts[bin_range] += 1
                    
                    # 统计每个桶中的像素数
                    self.binned_pixel_counts[bin_range] += gt_bin.numel()
    
    def get_results(self):
        """
        计算并返回最终的平均指标。
        返回:
            dict: 包含 'overall' 和 'binned' 结果的字典。
        """
        if self.num_samples == 0:
            print("Warning: No samples were processed.")
            return None

        # 计算整体平均值
        avg_overall_metrics = self.overall_metrics / self.num_samples
        overall_results = dict(zip(self.metric_names, avg_overall_metrics))
        
        # 计算分桶平均值和百分比
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
        """保存评估结果到文件。
        
        参数:
            filename (str): 保存文件的名称。默认为 "evaluation_results.txt"
        """
        import os
        
        results = self.get_results()
        if results is None:
            return

        # 确保保存目录存在
        os.makedirs(self.base_dir, exist_ok=True)
        filepath = os.path.join(self.base_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            header = " | ".join([f"{name:^10}" for name in self.metric_names])
            header += " | " + f"{'Percentage':^12} | {'PixelCount':^12}"
            separator = "-" * len(header)
            
            f.write(separator + "\n")
            f.write(f"{'Category':<15} | {header}\n")
            f.write(separator + "\n")

            # 写入整体结果
            overall = results['overall']
            metrics_str = " | ".join([f"{overall[name]:^10.4f}" for name in self.metric_names])
            f.write(f"{'Overall':<15} | {metrics_str} | {'100.00%':^12} | {results['total_pixels']:^12}\n")
            
            # 写入分桶结果
            binned = results['binned']
            for bin_range_str, bin_metrics in sorted(binned.items()):
                metrics_str = " | ".join([f"{bin_metrics[name]:^10.4f}" for name in self.metric_names])
                percentage_str = f"{bin_metrics['percentage']:^10.2f}%"
                pixel_count_str = f"{bin_metrics['pixel_count']:^12}"
                f.write(f"{bin_range_str:<15} | {metrics_str} | {percentage_str:^12} | {pixel_count_str}\n")
            
            f.write(separator + "\n")
        
        print(f"Results saved to: {filepath}")

    def display_results(self):
        """格式化并打印评估结果。"""
        results = self.get_results()
        if results is None:
            return

        header = " | ".join([f"{name:^10}" for name in self.metric_names])
        header += " | " + f"{'Percentage':^12} | {'PixelCount':^12}"
        print("-" * len(header))
        print(f"{'Category':<15} | {header}")
        print("-" * len(header))

        # 打印整体结果
        overall = results['overall']
        metrics_str = " | ".join([f"{overall[name]:^10.4f}" for name in self.metric_names])
        print(f"{'Overall':<15} | {metrics_str} | {'100.00%':^12} | {results['total_pixels']:^12}")
        
        # 打印分桶结果
        binned = results['binned']
        for bin_range_str, bin_metrics in sorted(binned.items()):
            metrics_str = " | ".join([f"{bin_metrics[name]:^10.4f}" for name in self.metric_names])
            percentage_str = f"{bin_metrics['percentage']:^10.2f}%"
            pixel_count_str = f"{bin_metrics['pixel_count']:^12}"
            print(f"{bin_range_str:<15} | {metrics_str} | {percentage_str:^12} | {pixel_count_str}")
        
        print("-" * len(header))


if __name__ == "__main__":
    # 假设你的测试循环是这样的
    # from your_model_file import Model
    # from your_dataloader_file import test_loader

    # model = Model()
    # model.eval()

    # 1. 在测试循环开始前，初始化评估器
    # 你可以自定义分桶范围和保存目录
    evaluator = DepthEvaluator(base_dir="./evaluation_results")

    # 2. 在测试循环中，为每个批次调用 update 方法
    # with torch.no_grad():
    #     for batch in test_loader:
    #         inputs, gt_depth = batch['image'], batch['depth']
    #         pred_depth = model(inputs)
            
    #         # 将数据传入评估器
    #         evaluator.update(gt_depth, pred_depth)

    # ------ 模拟测试循环 ------
    print("Simulating evaluation loop...")
    # 模拟10个批次的数据, batch_size=4, H=256, W=512
    for _ in range(10):
        # 生成随机的模拟数据
        # 真实深度在 [0.5, 4.5] 之间
        simulated_gt = torch.rand(4, 1, 256, 512) * 4.0 + 0.5
        # 模拟预测，带有一些随机噪声
        noise = torch.randn(4, 1, 256, 512) * 0.1
        simulated_pred = simulated_gt + noise
        simulated_pred = torch.clamp(simulated_pred, 0.5, 4.5) # 保证预测值在范围内
        
        # 调用 update
        evaluator.update(simulated_gt, simulated_pred)
    print("Simulation finished.")
    # -------------------------

    # 3. 在所有数据处理完毕后，获取并展示结果
    print("\n" + "="*20 + " Final Evaluation Results " + "="*20)
    # 你可以直接打印格式化的结果
    evaluator.display_results()
    
    # 保存结果到文件
    evaluator.save_results("depth_evaluation_results.txt")
