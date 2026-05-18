from models.PSNet import PSNet
from utils import set_seed
SEED_NUM = 0
set_seed(SEED_NUM)
# Add the missing ANTIALIAS attribute to PIL.Image
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
import argparse
import time
import csv

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import custom_transforms
from utils import tensor2array, cam_tensor2array, sonar_tensor2array, save_checkpoint, save_path_formatter, adjust_learning_rate
from loss_functions import compute_errors_train

from logger import TermLogger, AverageMeter
from itertools import chain
from tensorboardX import SummaryWriter
# from sequence_folders import SequenceFolder
from retrieve_dataset import RetrieveFolder
from tqdm import tqdm
from config.model_config import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('--data', metavar='DIR', default="./dataset/new/",help='path to dataset')
parser.add_argument('--data', metavar='DIR', default=DATA_DIR,help='path to dataset')
parser.add_argument('-j', '--workers', default=WORKERS, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('--epoch-size', default=500, type=int, metavar='N',
#                     help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')

parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                    metavar='N', help='mini-batch size')
# parser.add_argument('-b', '--batch-size', default=4, type=int,
#                     metavar='N', help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True,
                    help='evaluate model on validation set')

parser.add_argument('--pretrained-dps', dest='pretrained_dps', default=None, metavar='PATH', help='path to pre-trained dpsnet model')

parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full-train', default='progress_log_full_train.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-full-val', default='progress_log_full_val.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')

# parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--log-output', default=True, help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--ttype', default='../txt_files_filtered/train.txt', type=str, help='Text file indicates input data')
parser.add_argument('--ttype2', default='../txt_files_filtered/val_real.txt', type=str, help='Text file indicates input data')
parser.add_argument('--ttype3', default='../txt_files_filtered/val_sim.txt', type=str, help='Text file indicates input data')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=100)

# parser.add_argument('--label_factor', type=int ,default=1.08, help='label factors, depth of i th pseudo plane = label_factor**i')
# parser.add_argument('--nlabel', type=int ,default=32, help='number of label')
# parser.add_argument('--alpha', type=float ,default=np.deg2rad(30), help='angle of pseudo w.r.t the principal axis (degree)')
# parser.add_argument('--mindepth', type=float ,default=0.5, help='minimum depth')
parser.add_argument('--label_factor', type=int ,default=FACTOR, help='label factors, depth of i th pseudo plane = label_factor**i')
parser.add_argument('--nlabel', type=int ,default=N_LABEL, help='number of label')
parser.add_argument('--alpha', type=float ,default=np.deg2rad(ALPHA), help='angle of pseudo w.r.t the principal axis (degree)')
parser.add_argument('--mindepth', type=float ,default=MINDEPTH, help='minimum depth')

parser.add_argument('--maxdepth', type=float ,default=MAXDEPTH, help='minimum depth')

n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global n_iter
    args = parser.parse_args()

    ####################################################
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path/'train')
    output_writer = SummaryWriter(args.save_path/'valid')
    output_writer2 = SummaryWriter(args.save_path/'valid')
    ####################################################

    # Data loading code
    train_transform = custom_transforms.Compose([custom_transforms.ArrayToTensorGrey()])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensorGrey()])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = RetrieveFolder(
        args.data,
        transform=train_transform,
        seed=SEED_NUM,
        ttype=args.ttype
    )
    val_real_set = RetrieveFolder(
        args.data,
        transform=valid_transform,
        seed=SEED_NUM,
        ttype=args.ttype2
    )
    val_sim_set = RetrieveFolder(
        args.data,
        transform=valid_transform,
        seed=SEED_NUM,
        ttype=args.ttype3
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_real_set), len(val_real_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_sim_set), len(val_sim_set.scenes)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_real_loader = torch.utils.data.DataLoader(
        val_real_set, batch_size=8*args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_sim_loader = torch.utils.data.DataLoader(
        val_sim_set, batch_size=8*args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    nlabel = args.nlabel
    mindepth = args.mindepth
    label_factor = args.label_factor
    alpha = args.alpha
    
    dpsnet = PSNet(nlabel, mindepth, label_factor, alpha).to(device)

    if args.pretrained_dps and TRAIN_NEW:
        print("=> using pre-trained weights for PSNet (partially loading due to model changes)")

        # 1. Load the pretrained checkpoint.
        weights = torch.load(args.pretrained_dps)
        pretrained_dict = weights['state_dict']

        # 2. Keep only layers that do not start with the excluded prefixes.
        prefixes_to_exclude = ['rgb_feature_extraction.', 'sonar_feature_extraction.']
        
        # Filter the checkpoint dictionary.
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                        if not any(k.startswith(prefix) for prefix in prefixes_to_exclude)}

        # Report the filtering effect.
        original_keys_count = len(pretrained_dict)
        filtered_keys_count = len(filtered_dict)
        print(f"Original checkpoint has {original_keys_count} layers.")
        print(f"Excluding feature extractors, {filtered_keys_count} layers will be loaded.")

        # 3. Load the filtered weights. strict=False allows the changed feature
        # extractors to remain randomly initialized.
        missing_keys, unexpected_keys = dpsnet.load_state_dict(filtered_dict, strict=False)

        # Print the load status for verification.
        print(f"Successfully loaded model from: {args.pretrained_dps}")
        if unexpected_keys:
            print("\nWarning: The following keys from the checkpoint were not found in the model:")
            print(unexpected_keys)
        if missing_keys:
            print(f"\nThe following {len(missing_keys)} keys in the model were not found in the filtered checkpoint (as expected):")
            # Print only the first few examples.
            for k in sorted(missing_keys)[:5]:
                print(f"  - {k}")
            if len(missing_keys) > 5:
                print("  - ... and more")
                
        # Reinitialize the changed feature extraction networks explicitly.
        dpsnet.initialize_feature_extractors()

    elif args.pretrained_dps and not TRAIN_NEW:
        print("=> using pre-trained weights for DPSNet")
        weights = torch.load(args.pretrained_dps)
        dpsnet.load_state_dict(weights['state_dict'])
    
    else:
        print("=> no pre-trained weights provided, initializing all weights from scratch.")
        dpsnet.init_weights()
    
    cudnn.benchmark = True
    dpsnet = torch.nn.DataParallel(dpsnet)

    print('=> setting adam solver')

    parameters = chain(dpsnet.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss_real', 'validation_loss_sim'])

    with open(args.save_path/args.log_full_train, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss'])
        
    with open(args.save_path/args.log_full_val, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['n_iter', 'abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3'])
        
    # Create separate log file for real validation set
    log_full_val_real = args.log_full_val.replace('.csv', '_real.csv')
    with open(args.save_path/log_full_val_real, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['n_iter', 'abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3'])
        
    # Create separate log file for sim validation set  
    log_full_val_sim = args.log_full_val.replace('.csv', '_sim.csv')
    with open(args.save_path/log_full_val_sim, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['n_iter', 'abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3'])


    for epoch in range(args.epochs):
        print(f"EPOCH: {epoch}")
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train_loss = train(args, train_loader, dpsnet, optimizer, args.epoch_size, training_writer)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': dpsnet.module.state_dict()
            },
            epoch)
        
        # log_full_val_real = args.log_full_val.replace('.csv', '_real.csv')
        # log_full_val_sim = args.log_full_val.replace('.csv', '_sim.csv')
        
        errors_real, error_names = validate_with_gt(args, val_real_loader, dpsnet, epoch, output_writer, log_full_val_real)
        errors_sim, _ = validate_with_gt(args, val_sim_loader, dpsnet, epoch, output_writer2, log_full_val_sim)

        # error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))

        for error, name in zip(errors_real, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error_real = errors_real[0]
        decisive_error_sim = errors_sim[0]

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error_real, decisive_error_sim])

        print()


def train(args, train_loader, dpsnet: PSNet, optimizer, epoch_size, train_writer: SummaryWriter):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    # switch to train mode
    dpsnet.train()

    end = time.time()
    
    pbar = tqdm(enumerate(train_loader), total=min(len(train_loader), epoch_size), 
                desc=f"Training", unit="batch")
    
    for i, (cam_img, sonar_rect_img, depth_gt,\
                K, T_sonar_from_cam, T_cam,\
                distance_range, theta_range) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)
        cam_img_var = cam_img.to(device)
        sonar_rect_img_var = sonar_rect_img.to(device)
        depth_gt_var = depth_gt.to(device)
        K_var = K.to(device)
        T_sonar_from_cam_var = T_sonar_from_cam.to(device)
        distance_range_var = distance_range.to(device)
        theta_range_var = theta_range.to(device)


        # get mask
        mask = (depth_gt <= args.maxdepth) & (depth_gt >= args.mindepth) & (depth_gt == depth_gt)
        mask[:, :20] = False
        mask[:, -20:] = False

        mask.detach_()

        # depths = dpsnet(cam_img_var, sonar_rect_img_var, K_var, KT_inv_var, distance_range_var, theta_range_var)
        output_depths = dpsnet(cam_img_var, sonar_rect_img_var, K_var, T_sonar_from_cam_var, distance_range_var, theta_range_var)
        # output_depths include depth0, depth

        loss = 0.
        for l, depth in enumerate(output_depths):
            output = torch.squeeze(depth,1)
            loss += F.smooth_l1_loss(output[mask], depth_gt_var[mask], size_average=True) * pow(0.7, len(output_depths)-l-1)

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:

        #     train_writer.add_image('train rgb image input', tensor2array(cam_img[0]), n_iter, dataformats='HWC')
        #     train_writer.add_image('train sonar image input', sonar_tensor2array(sonar_rect_img[0]), n_iter, dataformats='HWC')
            
        #     depth_to_show = depth_gt_var.data[0].cpu()
        #     depth_to_show[depth_to_show > args.maxdepth] = 0
        #     train_writer.add_image('train Depth GT Normalized',
        #                            tensor2array(depth_to_show, max_value=args.maxdepth),
        #                            n_iter, dataformats='HWC')

        #     train_writer.add_image('train Depth Output Normalized',
        #                             tensor2array(depths[1].data[0].cpu(), max_value=args.maxdepth),
        #                             n_iter, dataformats='HWC')

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full_train, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item()])
      
        if i >= epoch_size - 1:
            break

        n_iter += 1
        
        pbar.set_postfix(loss=f"{losses.val[0]:.4f}", batch_time=f"{batch_time.val[0]:.3f}s")

    return losses.avg[0]


def validate_with_gt(args, val_loader, dpsnet, epoch, output_writer, log_file_name):
    batch_time = AverageMeter()
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))

    # switch to evaluate mode
    dpsnet.eval()

    end = time.time()
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")

        for i, (cam_img, sonar_rect_img, depth_gt,\
                K, T_sonar_from_cam, T_cam,\
                distance_range, theta_range) in pbar:
            
            # measure data loading time
            cam_img_var = cam_img.to(device)
            sonar_rect_img_var = sonar_rect_img.to(device)
            depth_gt_var = depth_gt.to(device)
            K_var = K.to(device)
            T_sonar_from_cam_var = T_sonar_from_cam.to(device)
            distance_range_var = distance_range.to(device)
            theta_range_var = theta_range.to(device)

            output_depth, variance = dpsnet(cam_img_var, sonar_rect_img_var, K_var, T_sonar_from_cam_var, distance_range_var, theta_range_var)

            mask = (depth_gt <= args.maxdepth-0.5) & (depth_gt >= args.mindepth) & (depth_gt == depth_gt)
            mask[:, :20] = False
            mask[:, -20:] = False
            mask.detach_()
            output = torch.squeeze(output_depth.data.cpu(),1)

            if i % 10 == 0: # write 10 validation result
                # val_iter = 100*epoch + i/10
                val_iter = i/10
                output_writer.add_image(f'{epoch} val rgb Input', cam_tensor2array(cam_img[0]), val_iter, dataformats='HW')
                output_writer.add_image(f'{epoch} val sonar Input', sonar_tensor2array(sonar_rect_img[0]), val_iter, dataformats='HWC')
                depth_to_show = depth_gt_var.data[0].cpu()
                depth_to_show[depth_to_show > args.maxdepth] = 0
                output_writer.add_image(f'{epoch} val target Depth Normalized', tensor2array(depth_to_show, max_value=args.maxdepth), val_iter, dataformats='HWC')

                output_writer.add_image(f'{epoch} val Depth Output', tensor2array(output_depth.data[0].cpu(), max_value=args.maxdepth), val_iter, dataformats='HWC')
          
                output_writer.flush()
                
            errors.update(compute_errors_train(depth_gt, output, mask))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            with open(args.save_path/log_file_name, 'a') as csvfile:
                line = f"{epoch}\t" + "\t".join([f"{x:.4f}" for x in errors.val]) + "\n"
                csvfile.write(line)
            
            pbar.set_postfix(abs_error=f"{errors.val[0]:.4f}")

    return errors.avg, error_names


if __name__ == '__main__':
    main()
