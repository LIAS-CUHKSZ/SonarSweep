from models.PSNet import PSNet

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

import custom_transforms
from utils import *
SEED_NUM = 0
set_seed(SEED_NUM)
# from loss_functions import compute_errors_test
from testing_metric import DepthEvaluator, compute_errors_test
from retrieve_dataset import RetrieveFolder, RetrieveFolderFull

import os
from path import Path

from config.model_config import MEAN, STD

def visualize_depth(depth_img, max_depth=5.0):
    """
    将深度图（通常是float类型）转换为可用于保存的可视化彩色图
    """
    # 裁剪到最大深度
    depth_in_metres = np.nan_to_num(depth_img, nan=max_depth)
    depth_in_metres[depth_in_metres > max_depth] = max_depth
    
    # 归一化到0-255
    # normalized_depth = cv2.normalize(depth_in_metres, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    normalized_depth = (depth_in_metres / max_depth * 255).astype(np.uint8)
    
    # 应用伪彩色映射
    colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    colored_depth[depth_in_metres == 0] = 0
    colored_depth[depth_in_metres == max_depth] = 0
    
    return colored_depth

def visualize_uncertainty(uncertainty_img, max_uncertainty=20.0):
    """
    将不确定性图（通常是float类型）转换为可用于保存的可视化彩色图
    """
    # 裁剪到最大不确定性
    uncertainty_in_metres = np.nan_to_num(uncertainty_img, nan=max_uncertainty)
    uncertainty_in_metres[uncertainty_in_metres > max_uncertainty] = max_uncertainty
    
    # 归一化到0-255
    normalized_uncertainty = (uncertainty_in_metres / max_uncertainty * 255).astype(np.uint8)
    
    # 应用伪彩色映射
    colored_uncertainty = cv2.applyColorMap(normalized_uncertainty, cv2.COLORMAP_JET)
    
    return colored_uncertainty


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')

parser.add_argument('--pretrained-dps', dest='pretrained_dps', default="./pretrained/045_025_B7.pth.tar", metavar='PATH', help='path to pre-trained dpsnet model')
# parser.add_argument('--pretrained-dps', dest='pretrained_dps', default="/home/clp/workspace/Sonar_sweep/checkpoints/vfov12hfov60/09-01-23:02/dpsnet_38_checkpoint.pth.tar", metavar='PATH', help='path to pre-trained dpsnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')


case = 'vfov12hfov60'
place = 'test'
# case = 'vfov12hfov60_real_Type-1C'
# parser.add_argument('--data', metavar='DIR', default=f"/media/clp/T9/SonarSweep_dataset/combined_and_enhanced_dataset/{case}",help='path to dataset')
parser.add_argument('--data', metavar='DIR', default=f"data/{case}",help='path to dataset')

parser.add_argument('--ttype', default=f'../{place}.txt', type=str, help='Text file indicates input data')
parser.add_argument('--output-dir', default=f'result/{case}/{place}', type=str, help='Output directory for saving predictions in a big 3D numpy file')

parser.add_argument('--label_factor', type=int ,default=1.05, help='label factors, depth of i th pseudo plane = label_factor**i')
parser.add_argument('--nlabel', type=int ,default=48, help='number of label')
parser.add_argument('--alpha', type=float ,default=np.deg2rad(45), help='angle of pseudo w.r.t the principal axis (degree)')
parser.add_argument('--mindepth', type=float ,default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float ,default=4.5, help='maximum depth')


# parser.add_argument('--output-print', action='store_true', help='print output depth')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency')

def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=> fetching scenes in '{}'".format(args.data))
    # normalize = custom_transforms.NormalizeCam(mean=MEAN, std=STD) # normalized_value = (x - mean) / std
    test_transform = custom_transforms.Compose([custom_transforms.ArrayToTensorGrey()])
    # TODO: 如果需要灰度图像转换为张量，可以使用以下代码
    # test_transform = custom_transforms.Compose([custom_transforms.ArrayToTensorGrey(), normalize])
    test_set = RetrieveFolderFull(
        args.data,
        seed=SEED_NUM,
        ttype=args.ttype,
        transform=test_transform,
    )

    print('{} samples found in {} valid scenes'.format(len(test_set), len(test_set.scenes)))
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    nlabel = args.nlabel
    mindepth = args.mindepth
    label_factor = args.label_factor
    dpsnet = PSNet(nlabel, mindepth, label_factor, args.alpha).to(device)
    
    pretrained_model_path = args.pretrained_dps
    weights = torch.load(pretrained_model_path)
    
    
    dpsnet.load_state_dict(weights['state_dict'])
    dpsnet.eval()
    print(f"Successfully load model from: {pretrained_model_path}")

    # output_dir= Path("result")
    output_dir= Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print_freq = args.print_freq
    errors = np.zeros((8, int(np.ceil(len(test_loader)/print_freq))), np.float32)
    

    evaluator = DepthEvaluator(base_dir=output_dir)

    # 创建进度条
    progress_bar = tqdm(enumerate(test_loader), 
                       total=len(test_loader), 
                       desc="Testing", 
                       unit="batch",
                       dynamic_ncols=True)
    
    with torch.no_grad():
        # for i, (cam_img, sonar_rect_img, depth_gt, K, KT_inv, distance_range, theta_range) in enumerate(test_loader):
        for i, (cam_img, sonar_rect_img, depth_gt,\
                K, T_sonar_from_cam, T_cam,\
                distance_range, theta_range,\
                cam_right_img, sonar_rect_original_img) in progress_bar:
               
            cam_img_var = cam_img.to(device)
            sonar_rect_img_var = sonar_rect_img.to(device)
            depth_gt_var = depth_gt.to(device)
            
            K_var = K.to(device)
            T_sonar_from_cam_var = T_sonar_from_cam.to(device)
            T_cam_var = T_cam.to(device)
            
            distance_range_var = distance_range.to(device)
            theta_range_var = theta_range.to(device)
            
            
            start = time.time()
            output_depth, variance = dpsnet(cam_img_var, sonar_rect_img_var, K_var, T_sonar_from_cam_var, distance_range_var, theta_range_var)
            
            elps = time.time() - start
        
            # mask = (depth_gt <= args.maxdepth) & (depth_gt >= args.mindepth) & (depth_gt == depth_gt) # tgt_depth == tgt_depth: 排除NaN值(NaN不等于自身)
            # mask[:, :20] = False  # 前10列设置为False
            # mask[:, -20:] = False

            output_depth = torch.squeeze(output_depth.data.cpu(),1)
            uncertainty_array = torch.squeeze(variance.data.cpu(),1)

            # 检查mask是否有有效值，避免空张量导致的错误
            evaluator.update(depth_gt, output_depth)

            errors[:,i] = compute_errors_test(depth_gt, output_depth)
            # 更新进度条显示当前错误信息
            progress_bar.set_postfix({
                'Abs_Rel': f'{errors[0,i]:.4f}',
                'RMSE': f'{errors[3,i]:.4f}',
                'Time': f'{elps:.3f}s'
            })
        

            # print('Elapsed Time {}  Abs rel Error {:.4f}'.format(elps, errors[1,i]))

            output_depth_array = (output_depth).numpy()[0]
            uncertainty_array = (uncertainty_array).numpy()[0]
            gt_depth_array = (depth_gt).numpy()[0]
            
            cam_img_array = cam_tensor2array(cam_img[0])
            cam_img_array = (cam_img_array * 255).astype(np.uint8)
        
            # # 保存
            # if errors[0,i] > 0.1 and False:
            if True:
                os.makedirs(output_dir / f'{i:04d}', exist_ok=True)
                

                cv2.imwrite(str(output_dir / f'{i:04d}' / 'cam_image_full.png'), cam_right_img.squeeze().numpy())
        
                sonar_image = sonar_rect_original_img.squeeze().numpy()
                cv2.imwrite(str(output_dir / f'{i:04d}' / 'sonar_image_full.png'), sonar_image)
                cv2.imwrite(str(output_dir / f'{i:04d}' / 'sonar_image.png'), sonar_image)

                cv2.imwrite(str(output_dir / f'{i:04d}' / 'cam_image.png'), cam_img_array)

                # np.save(output_dir / f'{i:04d}' / 'cam_right_pose.txt', T_cam)
                np.savetxt(output_dir / f'{i:04d}' / 'cam_right_pose.txt', T_cam.squeeze(axis=0), fmt='%.6f')

                np.save(output_dir / f'{i:04d}' / 'output_depth.npy', output_depth_array)
                np.save(output_dir / f'{i:04d}' / 'uncertainty.npy', uncertainty_array)
                cv2.imwrite(str(output_dir / f'{i:04d}' / 'output_depth.png'), visualize_depth(output_depth_array))            
                cv2.imwrite(str(output_dir / f'{i:04d}' / 'uncertainty.png'), visualize_uncertainty(uncertainty_array))            
                
                np.save(output_dir/ f'{i:04d}' / 'gt_depth.npy', gt_depth_array)
                cv2.imwrite(str(output_dir / f'{i:04d}' / 'gt_depth.png'), visualize_depth(gt_depth_array))   
    
    # 关闭进度条
    progress_bar.close()

    print("\n" + "="*20 + " Final Evaluation Results " + "="*20)
    # 你可以直接打印格式化的结果
    evaluator.display_results()
    
    # 保存结果到文件
    evaluator.save_results("depth_evaluation_results.txt")

    # mean_errors = errors.mean(-1)
    # error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3']
    # print("{}".format(args.output_dir))
    # print("Depth Results : ")
    # print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    # print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors))
    # np.savetxt(output_dir/'errors.csv', mean_errors, fmt='%1.4f', delimiter=',')

    error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3']
    # 使用nanmean来正确处理NaN值（当某些样本没有有效深度值时）
    mean_errors = np.nanmean(errors, axis=1)  # 沿第一个维度计算平均值，忽略NaN

    # 打印结果
    print("{}".format(args.output_dir))
    print("Depth Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors))

    # 打印有效样本数量
    valid_samples = np.sum(~np.isnan(errors[0, :]))  # 计算第一个错误指标中非NaN的数量
    total_samples = errors.shape[1]
    print(f"Valid samples: {valid_samples}/{total_samples}")

    # 保存错误数据到CSV
    # 创建一个新数组，第一行是平均值，其余行是每个样本的完整错误
    full_errors = np.vstack([mean_errors, errors.T])

    # 创建带有注释的CSV标题
    header = "# First row: mean errors (nanmean), remaining rows: individual sample errors\n"
    header += f"# Valid samples: {valid_samples}/{total_samples}\n"
    header += ','.join(error_names)

    # 保存到CSV文件
    np.savetxt(output_dir/'errors.csv', full_errors, fmt='%1.4f', delimiter=',', 
            header=header, comments='')
    


if __name__ == '__main__':
    main()
