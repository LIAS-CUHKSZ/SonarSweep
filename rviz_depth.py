#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import PointCloud2, PointField, Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from pathlib import Path
import csv
from loss_functions import compute_errors_test, compute_errors_train
import torch


DEPTH_MIN = 0.5
DEPTH_MAX = 5

def visualize_uncertainty(uncertainty_img, mask, max_uncertainty=2.0):
    """将不确定性图转换为可视化彩色图"""
    uncertainty_in_metres = np.nan_to_num(uncertainty_img, nan=max_uncertainty)
    uncertainty_in_metres[uncertainty_in_metres > max_uncertainty] = max_uncertainty
    uncertainty_in_metres[~mask] = 0
    
    normalized_uncertainty = (uncertainty_in_metres / max_uncertainty * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized_uncertainty, cv2.COLORMAP_JET)

class DataPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('data_publisher', anonymous=True)
        
        # Create publishers for point clouds
        self.pubs = {
            'est': rospy.Publisher('/depth_cloud_est', PointCloud2, queue_size=1),
            'gt': rospy.Publisher('/depth_cloud_gt', PointCloud2, queue_size=1)
        }
        
        # Create publishers for images
        self.image_pubs = {
            'rgb': rospy.Publisher('/rgb_image', Image, queue_size=1),
            'sonar': rospy.Publisher('/sonar_image', Image, queue_size=1),
            'uncertainty': rospy.Publisher('/uncertainty_image', Image, queue_size=1),
            'error_image': rospy.Publisher('/error_image', Image, queue_size=1)
        }
        
        # Create frames dictionary
        self.tfs = {
            'est': 'camera_link_est',
            'gt': 'camera_link_gt'
        }
        
        # Create bridge for converting OpenCV images to ROS messages
        self.bridge = CvBridge()
        
        # Create TF broadcaster
        self.broadcaster = StaticTransformBroadcaster()
        
        # Publish static TF transformations
        self.publish_static_tfs()
        
        # Set publishing rate
        self.rate = rospy.Rate(100)
        self.results = []

    def publish_static_tfs(self):
        """Publish static TF transformations for both estimated and ground truth frames"""
        static_transforms = []
        
        for name, frame_id in self.tfs.items():
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "map"
            transform.child_frame_id = frame_id

            transform.transform.translation.x = 0.0
            transform.transform.translation.y = 0.0 
            transform.transform.translation.z = 1.5 if name == 'est' else 0.5

            transform.transform.rotation.x = np.sin(-np.pi/4)
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = np.cos(-np.pi/4)
            
            static_transforms.append(transform)

        self.broadcaster.sendTransform(static_transforms)

    def depth_to_pointcloud(self, depth_data, frame_id, valid_mask):
        """Convert depth image to point cloud data"""
        rows, cols = depth_data.shape
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Camera intrinsics
        fx = fy = 320.0
        cx = cols/2
        cy = rows/2

        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]

        x = np.where(valid_mask, (u - cx) * depth_data / fx, 0)
        y = np.where(valid_mask, (v - cy) * depth_data / fy, 0)
        z = np.where(valid_mask, depth_data, 0)

        normalized_depth = np.clip(depth_data / np.nanmax(depth_data), 0, 1)
        valid_colors = (normalized_depth[valid_mask] * 255).astype(np.uint8)
        
        rgb_packed = (valid_colors.astype(np.uint32) << 16 |
                    ((255 - valid_colors).astype(np.uint32)) |
                    (np.full_like(valid_colors, 255, dtype=np.uint32) << 24))
        
        points = np.column_stack((
            x[valid_mask],
            y[valid_mask],
            z[valid_mask],
            rgb_packed
        ))
        return pc2.create_cloud(header, fields, points)

    def publish_data(self, folder_path):
        """Publish all data from a specific folder"""
        # Load depth data
        est_depth = np.load(folder_path / 'output_depth.npy')
        gt_depth = np.load(folder_path / 'gt_depth.npy')
        uncertainty_array = np.load(folder_path / 'uncertainty.npy')    
        
        # Create valid masks
        valid_mask_gt = ~np.isnan(gt_depth) & (gt_depth > DEPTH_MIN) & (gt_depth < DEPTH_MAX)
        valid_mask_gt[:, :20] = False
        valid_mask_gt[:, -20:] = False
        valid_mask_gt[:4, :] = False  # 过滤上边缘20像素
        valid_mask_gt[-4:, :] = False  # 过滤下边缘20像素

        valid_mask = valid_mask_gt
      
        valid_mask_est = ~np.isnan(est_depth) & (est_depth > DEPTH_MIN) & (est_depth < DEPTH_MAX)
        valid_mask_est[:, :20] = False
        valid_mask_est[:, -20:] = False
        valid_mask_est[:4, :] = False  # 过滤上边缘20像素
        valid_mask_est[-4:, :] = False  # 过滤下边缘20像素

        print(f"Valid pixels: {np.sum(valid_mask)}/{valid_mask.size}")
        
        # Calculate absolute error for visualization
        error = np.abs(est_depth - gt_depth)
        abs_error_full = np.zeros_like(gt_depth, dtype=np.float32)
        abs_error_full[valid_mask] = error[valid_mask]
        normalized_error = np.clip(abs_error_full, 0, 1.0)
        
        abs_error_img = np.zeros_like(gt_depth, dtype=np.uint8)
        abs_error_img[valid_mask] = (normalized_error[valid_mask] * 255).astype(np.uint8)
        abs_error_norm_color = cv2.applyColorMap(abs_error_img, cv2.COLORMAP_VIRIDIS)

        # Load images
        rgb_img = cv2.imread(str(folder_path / 'cam_image.png'))
        sonar_img = cv2.imread(str(folder_path / 'sonar_image.png'))
        sonar_img = cv2.flip(sonar_img, 0)

        uncertainty_img = cv2.imread(str(folder_path / 'uncertainty.png'))

        # Publish point clouds
        self.pubs['est'].publish(self.depth_to_pointcloud(est_depth, self.tfs['est'], valid_mask_est))
        self.pubs['gt'].publish(self.depth_to_pointcloud(gt_depth, self.tfs['gt'], valid_mask_gt))
        
        
        # Convert to torch tensors and apply mask directly (like in test.py)
        gt_tensor = torch.from_numpy(gt_depth).float().unsqueeze(0)  # [1, H, W]
        pred_tensor = torch.from_numpy(est_depth).float().unsqueeze(0)  # [1, H, W]

        metrics = compute_errors_test(gt_tensor, pred_tensor)
        abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3 = metrics
    
        print(f"Performance metrics - Abs rel: {abs_rel:.4f}, Abs diff: {abs_diff:.4f}, Sq rel: {sq_rel:.4f}")
        
        # Store results for CSV
        result_dict = {
            'folder': folder_path.name,
            'abs_diff': abs_diff,
            'abs_rel': abs_rel, 
            'sq_rel': sq_rel
        }
        
        if 'rmse' in locals():
            result_dict.update({
                'rmse': rmse, 'rmse_log': rmse_log,
                'a1': a1, 'a2': a2, 'a3': a3
            })
        
        self.results.append(result_dict)
        
        # Publish images
        timestamp = rospy.Time.now()
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
        error_image_msg = self.bridge.cv2_to_imgmsg(abs_error_norm_color, "bgr8")
        sonar_msg = self.bridge.cv2_to_imgmsg(sonar_img, "bgr8")
        uncertainty_msg = self.bridge.cv2_to_imgmsg(uncertainty_img, "bgr8")
        
        for msg in [rgb_msg, sonar_msg, uncertainty_msg, error_image_msg]:
            msg.header.stamp = timestamp
            
        self.image_pubs['rgb'].publish(rgb_msg)
        self.image_pubs['sonar'].publish(sonar_msg)
        self.image_pubs['uncertainty'].publish(uncertainty_msg)
        self.image_pubs['error_image'].publish(error_image_msg)

        print(f"Published data from folder: {folder_path}")

    def save_results_to_csv(self, csv_filename="summary.csv"):
        """Save all results to CSV file and calculate averages"""
        if not self.results:
            print("No results to save")
            return
            
        base_fieldnames = ['folder', 'abs_diff', 'abs_rel', 'sq_rel']
        additional_fieldnames = ['rmse', 'rmse_log', 'a1', 'a2', 'a3']
        
        has_additional_metrics = any(all(key in result for key in additional_fieldnames) 
                                   for result in self.results)
        
        fieldnames = base_fieldnames + (additional_fieldnames if has_additional_metrics else [])
            
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                filtered_result = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(filtered_result)
            
            if len(self.results) > 0:
                valid_results = [result for result in self.results 
                               if not any(np.isnan(result.get(key, np.nan)) 
                                         for key in ['abs_diff', 'abs_rel', 'sq_rel'])]
                
                if len(valid_results) > 0:
                    avg_data = {'folder': 'AVERAGE'}
                    
                    for field in fieldnames[1:]:
                        if field in ['abs_diff', 'abs_rel', 'sq_rel']:
                            avg_data[field] = sum(r[field] for r in valid_results) / len(valid_results)
                        elif has_additional_metrics and field in additional_fieldnames:
                            values = [r.get(field, np.nan) for r in valid_results 
                                    if not np.isnan(r.get(field, np.nan))]
                            avg_data[field] = sum(values) / len(values) if values else float('nan')
                    
                    writer.writerow(avg_data)
                    
                    print(f"Results saved to {csv_filename}")
                    print(f"Valid samples: {len(valid_results)}/{len(self.results)}")
                    print(f"Average metrics - Abs diff: {avg_data['abs_diff']:.4f}, "
                          f"Abs rel: {avg_data['abs_rel']:.4f}, Sq rel: {avg_data['sq_rel']:.4f}")
                else:
                    print(f"No valid results found for averaging (all contain NaN values)")
                    nan_data = {'folder': 'AVERAGE'}
                    for field in fieldnames[1:]:
                        nan_data[field] = 'NaN'
                    writer.writerow(nan_data)

def main():
    try:
        data_publisher = DataPublisher()

        result_dir = Path("result/vfov12hfov60/test")   
        csv_filename = f"summary_{result_dir.name}.csv"
        
        while not rospy.is_shutdown():
            for folder in sorted(result_dir.glob("[0-9]*")):
                if rospy.is_shutdown():
                    break
                
                if folder.name.isdigit():
                    rospy.loginfo(f"Publishing data from folder: {folder}")
                    data_publisher.publish_data(folder)
                    input()
            
            rospy.loginfo("Completed one round of publishing, waiting before next round...")
            rospy.sleep(2.0)
            break
            
    except rospy.ROSInterruptException:
        print("ROS interrupted, saving results...")
        data_publisher.save_results_to_csv(csv_filename)
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Saving results before exit...")
        data_publisher.save_results_to_csv(csv_filename)
        raise

if __name__ == '__main__':
    main()