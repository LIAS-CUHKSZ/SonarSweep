from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
import random
import os
import cv2
from config.model_config import MEAN, STD

def set_seed(seed=42):
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)  # Python randomness
    np.random.seed(seed)  # NumPy randomness
    torch.manual_seed(seed)  # Torch CPU randomness
    torch.cuda.manual_seed(seed)  # Torch GPU randomness
    torch.cuda.manual_seed_all(seed)  # Multi-GPU randomness
    
    # Use a single CPU thread to reduce nondeterminism.
    torch.set_num_threads(1)
    
    # cuDNN settings for deterministic behavior.
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed.
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    if max_value is None:
        max_value = tensor.max()
    try:
        color_cvt = cv2.COLOR_BGR2RGB
        if colormap == 'rainbow':
            colormap = cv2.COLORMAP_RAINBOW
        elif colormap == 'bone':
            colormap = cv2.COLORMAP_BONE
        array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
        colored_array = cv2.applyColorMap(array, colormap)
        array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        #array = array.transpose(2, 0, 1)
    except ImportError:
        if tensor.ndimension() == 2:
            tensor.unsqueeze_(2)
        array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)
    return array

def cam_tensor2array(tensor):
    # if tensor.ndimension() == 3:
    #     array = MEAN + tensor.numpy().transpose(1,2,0)*STD
    # else: 
    array = MEAN + tensor.squeeze().numpy()*STD
    return array

def sonar_tensor2array(tensor, max_value=1.0):
    """
    Convert a single-channel tensor to a grayscale image array for TensorBoard.
    """
    if max_value is None:
        max_value = tensor.max()
    
    # Single-channel sonar image.
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        # Squeeze and normalize to the 0-1 range.
        array = tensor.squeeze().numpy() / max_value
        array = array.clip(0, 1)
        
        # Convert a single-channel image to three-channel grayscale in HWC format.
        # TensorBoard expects RGB images, even for grayscale content.
        h, w = array.shape
        rgb_array = np.zeros((h, w, 3), dtype=np.float32)
        rgb_array[:, :, 0] = array
        rgb_array[:, :, 1] = array
        rgb_array[:, :, 2] = array
        
        return rgb_array
    
    # RGB image.
    elif tensor.ndimension() == 3 and tensor.size(0) == 3:
        array = 0.5 + tensor.numpy().transpose(1, 2, 0) * 0.5
        return array
        
    return None


def save_checkpoint(save_path, dpsnet_state, epoch, filename='checkpoint.pth.tar'):
    file_prefixes = ['dpsnet']
    states = [dpsnet_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}_{}'.format(prefix,epoch,filename))


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
