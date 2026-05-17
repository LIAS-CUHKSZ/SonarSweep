from __future__ import division
import torch
import numpy as np

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb_img, sonar_rect_img):
        for t in self.transforms:
            rgb_img, sonar_rect_img = t(rgb_img, sonar_rect_img)
        return rgb_img, sonar_rect_img


class Normalize(object):
    def __init__(self, mean, std, gamma=0.3):
        self.mean = mean
        self.std = std
        self.gamma = gamma

    def __call__(self, cam_tensor, sonar_rect_tensor):
        cam_tensor_norm = cam_tensor.sub_(self.mean).div_(self.std)
        sonar_rect_tensor_gamma = torch.pow(sonar_rect_tensor, self.gamma)
        return cam_tensor_norm, sonar_rect_tensor_gamma


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, rgb_img, sonar_rect_img):
        # put it from HWC to CHW format
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        # handle numpy array
        cam_tensor = torch.from_numpy(rgb_img).float()/255
        sonar_rect_tensor = torch.from_numpy(sonar_rect_img).float()/255
       
        return cam_tensor, sonar_rect_tensor

class NormalizeCam(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, cam_tensor, sonar_rect_tensor):
        cam_tensor_norm = cam_tensor.sub_(self.mean).div_(self.std)
        return cam_tensor_norm, sonar_rect_tensor
    
class ArrayToTensorGrey(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, rgb_img, sonar_rect_img):
        # handle numpy array
        cam_tensor = torch.from_numpy(rgb_img).float()/255
        sonar_rect_tensor = torch.from_numpy(sonar_rect_img).float()/255
       
        return cam_tensor, sonar_rect_tensor
