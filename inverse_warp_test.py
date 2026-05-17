from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
pixel_coords = None

import custom_transforms
from models.inverse_warp import warp_camera_to_sonar
# from sequence_folders import SequenceFolder
from retrieve_dataset import RetrieveFolder
import argparse
import cv2
import os


def show_resized_image(title, image, scale=3):
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(title, resized)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--data', metavar='DIR', default="./dataset/test/",help='path to dataset')
    parser.add_argument('--data', metavar='DIR', default="/home/clp/workspace/Sonar_sweep/dataset/real/",help='path to dataset')
    parser.add_argument('--ttype', default='test.txt', type=str, help='Text file indicates input data')
  
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size')

    parser.add_argument('--output-dir', default='result/test_real', type=str, help='Output directory for saving predictions in a big 3D numpy file')


    args = parser.parse_args()
    # normalize = custom_transforms.Normalize(mean=0.5, std=0.5, gamma=0.3) # normalized_value = (x - mean) / std
    # test_transform = custom_transforms.Compose([custom_transforms.ArrayToTensorGrey(), normalize])
    test_transform = custom_transforms.Compose([custom_transforms.ArrayToTensorGrey()])
    test_set = RetrieveFolder(
        args.data,
        seed=1,
        ttype=args.ttype,
        transform=test_transform,
    )
    dir_path = args.output_dir
    os.makedirs(dir_path, exist_ok=True)


    print('{} samples found in {} valid scenes'.format(len(test_set), len(test_set.scenes)))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    
    device = 'cuda'

    # depths = torch.tensor([0.5*1.08**i for i in range(32)]) # 0.5~5.3 30
    
    # depths = torch.tensor([0.3*1.05**i for i in range(64)]) 
    depths = torch.tensor([0.3*1.1**i for i in range(32)]) 
    # depths = torch.tensor([0.75*1.03**i for i in range(64)]) # 0.75~4.973 60
    alpha = torch.tensor(np.deg2rad(45))
    with torch.no_grad():
        for img_index, (rgb_img, sonar_rect_img, depth_gt,\
                K, T_sonar_from_cam, T_cam,\
                distance_range, theta_range) in enumerate(test_loader):
            
            rgb_img_var = rgb_img.to(device)
            sonar_rect_img_var = sonar_rect_img.to(device)
            depth_gt_var = depth_gt.to(device)
            K_var = K.to(device)
            T_sonar_from_cam = T_sonar_from_cam.to(device)
            distance_range_var = distance_range.to(device)
            theta_range_var = theta_range.to(device)
            
            rgb_img = rgb_img.cpu().numpy().squeeze()
            # rgb_img = np.transpose(rgb_img, (1, 2, 0))
            
            # show_resized_image(f"RGB Image {img_index}", (255*rgb_img).astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            for depth_index, depth in enumerate(depths):
                # warped_sonar_feature = warp_differentiable(K_var, KT_inv_var, rgb_img_feature.size(), sonar_feature, depth, 
                #                                             distance_range_var, theta_range_var, self.alpha)
                scale_factor = 1
                K_cam_scaled = K_var.clone()
                K_cam_scaled[:, :2, :] /= scale_factor  # Scale down the focal lengths
                target_image_shape = (int(rgb_img_var.size()[-2]/scale_factor), int(rgb_img_var.size()[-1]/scale_factor))
                                
                warped_sonar_feature = warp_camera_to_sonar(sonar_rect_img_var, K_cam_scaled, T_sonar_from_cam, target_image_shape, 
                                                            depth, distance_range_var, theta_range_var, alpha)
                # warped_sonar_feature = warp_camera_to_sonar(sonar_rect_img_var, K_var, T_sonar_from_cam, rgb_img_var.size(), 
                #                                             depth, distance_range_var, theta_range_var, alpha)
                output_first_image = warped_sonar_feature[0]
                output_numpy = output_first_image.detach().cpu().numpy() # Shape: (1, 480, 640)
                output_transposed = np.transpose(output_numpy, (1, 2, 0)) # Shape: (480, 640, 1)
                output_squeezed = output_transposed.squeeze()
                
                output_uint8 = (output_squeezed * 255).astype(np.uint8)
                cv2.imwrite(f"{dir_path}/image{img_index}_{depth_index}_{depth}m.png", output_uint8)
                # show_resized_image(f"{depth}m.png", output_uint8, scale=5)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
