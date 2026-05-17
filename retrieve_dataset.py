import torch.utils.data as data
import numpy as np
# from scipy.misc import imread
import imageio

from path import Path
import random
import cv2

# def load_as_float(path):
#     return imread(path).astype(np.float32)

def load_as_float(path):
    return imageio.imread(path).astype(np.float32)

class RetrieveFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, ttype='train.txt', transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/ttype
        # scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        scenes = [self.root/folder.rstrip() for folder in open(scene_list_path)]
        self.scenes = sorted(scenes, key=lambda s: int(str(s).split('_')[-1]))
        self.ttype = ttype
        self.transform = transform
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []

        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cropped_cam_intrinsic.txt').astype(np.float32).reshape((3, 3))
            T_cam = np.genfromtxt(scene/'cam_right_pose.txt').astype(np.float32).reshape((4, 4))
            T_sonar_from_cam = np.genfromtxt(scene/'T_camright2sonar.txt').astype(np.float32).reshape((4, 4))
            distance_range,_,_,theta_range,_,_,_ = np.genfromtxt(scene/'sonar_intrinsic.txt')
            # TODO
            # cam_img = scene.files('cropped_cam_right.png')[0]
            cam_img = scene.files('enhanced_gray_cam_right.png')[0]
            
            sonar_rect_img = scene.files('sonar_rect_denoise.png')[0]
            depth_gt = scene.files('cropped_depth_right.npy')[0]
            
            sample = {
                    'cam_img': cam_img, 'sonar_rect_img': sonar_rect_img, 'depth_gt': depth_gt,
                    'intrinsics': intrinsics, 'T_sonar_from_cam': T_sonar_from_cam,
                    'T_cam': T_cam,
                    'distance_range': distance_range, 'theta_range': theta_range
                    }
            sequence_set.append(sample)

        if self.ttype == 'train.txt':
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        
        cam_img = load_as_float(sample['cam_img'])
        sonar_rect_img = load_as_float(sample['sonar_rect_img'])
        depth_gt = np.load(sample['depth_gt'])
        K = np.copy(sample['intrinsics'])
        # KT_inv = np.linalg.inv(K.T)
        T_sonar_from_cam = sample['T_sonar_from_cam']
        T_cam = sample['T_cam']
        distance_range = np.copy(sample['distance_range'])
        theta_range = np.copy(sample['theta_range'])
        
        if self.transform is not None:
            cam_img, sonar_rect_img = self.transform(cam_img, sonar_rect_img)

        # TODO: if u use grey images, you need to change the shape of cam_img
        return cam_img.unsqueeze(0), sonar_rect_img.unsqueeze(0), depth_gt, \
                K, T_sonar_from_cam, T_cam, \
                distance_range.reshape(1), theta_range.reshape(1)    # np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
    

class RetrieveFolderFull(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, ttype='train.txt', transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/ttype
        # scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        scenes = [self.root/folder.rstrip() for folder in open(scene_list_path)]
        self.scenes = sorted(scenes, key=lambda s: int(str(s).split('_')[-1]))
        self.ttype = ttype
        self.transform = transform
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []

        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cropped_cam_intrinsic.txt').astype(np.float32).reshape((3, 3))
            T_cam = np.genfromtxt(scene/'cam_right_pose.txt').astype(np.float32).reshape((4, 4))
            T_sonar_from_cam = np.genfromtxt(scene/'T_camright2sonar.txt').astype(np.float32).reshape((4, 4))
            distance_range,_,_,theta_range,_,_,_ = np.genfromtxt(scene/'sonar_intrinsic.txt')
            # TODO
            # cam_img = scene.files('cropped_cam_right.png')[0]
            cam_img = scene.files('enhanced_gray_cam_right.png')[0]
            
            sonar_rect_img = scene.files('sonar_rect_denoise.png')[0]
            depth_gt = scene.files('cropped_depth_right.npy')[0]
            
            # 添加cam_right.png和sonar_rect.png的路径
            cam_right_img = scene.files('cam_right.png')[0]
            sonar_rect_original_img = scene.files('sonar_rect.png')[0]
            
            sample = {
                    'cam_img': cam_img, 'sonar_rect_img': sonar_rect_img, 'depth_gt': depth_gt,
                    'cam_right_img': cam_right_img, 'sonar_rect_original_img': sonar_rect_original_img,
                    'intrinsics': intrinsics, 'T_sonar_from_cam': T_sonar_from_cam,
                    'T_cam': T_cam,
                    'distance_range': distance_range, 'theta_range': theta_range
                    }
            sequence_set.append(sample)

        if self.ttype == 'train.txt':
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        
        cam_img = load_as_float(sample['cam_img'])
        sonar_rect_img = load_as_float(sample['sonar_rect_img'])
        depth_gt = np.load(sample['depth_gt'])
        
        # 读取新添加的图像文件
        
        K = np.copy(sample['intrinsics'])
        # KT_inv = np.linalg.inv(K.T)
        T_sonar_from_cam = sample['T_sonar_from_cam']
        T_cam = sample['T_cam']
        distance_range = np.copy(sample['distance_range'])
        theta_range = np.copy(sample['theta_range'])
        
        if self.transform is not None:
            cam_img, sonar_rect_img = self.transform(cam_img, sonar_rect_img)

        cam_right_img = cv2.imread(sample['cam_right_img'])
        sonar_rect_original_img = cv2.imread(sample['sonar_rect_original_img'])
        
        # TODO: if u use grey images, you need to change the shape of cam_img
        return cam_img.unsqueeze(0), sonar_rect_img.unsqueeze(0), depth_gt, \
                K, T_sonar_from_cam, T_cam, \
                distance_range.reshape(1), theta_range.reshape(1), \
                cam_right_img, sonar_rect_original_img    # np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)