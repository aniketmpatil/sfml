import torch
import time
import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn

from IPython import embed
from datasets.sequence_folders import SequenceFolder, ValidationSetWithPose
import custom_transforms as cust_trans
import models
import csv
import numpy as np
from inverse_warp import pose_vec2mat
from utils import save_checkpoint, save_path_formatter, log_output_tensorboard, tensor2array
from loss_functions import compute_depth_errors, smooth_loss, explainability_loss, photometric_reconstruction_loss, compute_pose_error
from logger import AverageMeter
from tensorboardX import SummaryWriter
from IPython import embed

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
# parser.add_argument('--dataset-format', default='sequential', metavar='STR',
#                     help='dataset format, stacked: stacked frames (from original TensorFlow code) '
#                     'sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use depth ground truth for validation. '
                    'You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--with-pose', action='store_true', help='use pose ground truth for validation. '
                    'You need to store it in text files of 12 columns see data/kitti_raw_loader.py for an example '
                    'Note that for kitti, it is recommend to use odometry train set to test pose')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',         #
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',                   #
                    help='evaluate model on validation set')
# parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
#                     help='path to pre-trained dispnet model')
# parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
#                     help='path to pre-trained Exp Pose net model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')    #
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
# parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('-f', '--training-output-freq', type=int,
                    help='frequence for outputting dispnet outputs and warped imgs at training for all scales. '
                         'if 0, will not output',
                    metavar='N', default=0)
parser.add_argument('-eps', '--uncert-loss-epsilon', type=float, help='epsilon for softplus function', metavar='EP', default=0.001)
parser.add_argument('--uncert', action='store_true', help='use uncertainity based loss function for photometric loss')

if torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")
best_error = -1
n_iter = 0

def main():
    ## SETUP
    global device, best_error, n_iter
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # if args.evaluate:
    #     args.epochs = 0

    ## Save_Path for checkpoints
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print('Checkpoints will be saved to save_path: {}'.format(args.save_path))
    args.save_path.makedirs_p()

    ## TensorBoard Writer
    tb_writer = SummaryWriter(args.save_path)

    # Custom transforms for train and validation sets
    normalize = cust_trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = cust_trans.Compose([
        cust_trans.RandomHorizontalFlip(),
        cust_trans.RandomScaleCrop(),
        cust_trans.ArrayToTensor(),
        normalize
    ])
    valid_transform = cust_trans.Compose([
        cust_trans.ArrayToTensor(),
        normalize
    ])

    # Get sequences from train list
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    # Get sequences from validation list
    if args.with_pose and args.with_gt:
        val_set = val_set = ValidationSetWithPose(
            args.data,
            sequence_length=args.sequence_length,
            transform=valid_transform
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
        )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    
    # Data Loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # Create Models - DispNetS and PoseExpNet
    disp_net = models.DispNetS().to(device)
    # pose_exp_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp = False).to(device)
    pose_exp_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp = args.mask_loss_weight > 0).to(device)

    # Weights Initialization
    disp_net.init_weights()       # Code to use pretrained weights also exists
    pose_exp_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)
    print("Number of GPUs Available: ", torch.cuda.device_count())
    
    # torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count(), init_method='...')
    # disp_net = torch.nn.parallel.DistributedDataParallel(disp_net)
    # pose_exp_net = torch.nn.parallel.DistributedDataParallel(pose_exp_net)

    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_exp_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(
        optim_params,
        betas=(args.momentum, args.beta),
        weight_decay=args.weight_decay
    )

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

    if args.evaluate:
        if args.with_gt and args.with_pose:
            errors, error_names = validate_with_gt_pose(args, val_loader, disp_net, pose_exp_net, 0, tb_writer)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net, 0, tb_writer)
        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, 0)

    for epoch in range(args.epochs):
        print("Epoch :", epoch)

        train_loss = train(args, train_loader, disp_net, pose_exp_net, optimizer, args.epoch_size, epoch, tb_writer)

        if args.with_gt and args.with_pose:
            # Validating using pose GT and generated depth from velodyne points
            errors, error_names = validate_with_gt_pose(args, val_loader, disp_net, pose_exp_net, epoch, tb_writer)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, tb_writer)

        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, epoch)
        
        decisive_error = errors[1]      # Choose which error measures model performance
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path,
            {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            },
            {
                'epoch': epoch + 1,
                'state_dict': pose_exp_net.module.state_dict()
            },
            is_best
        )

def train(args, train_loader, disp_net, pose_exp_net, optimizer, epoch_size, epoch, tb_writer):
    global device, n_iter
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight
    
    # Train mode
    disp_net.train()
    pose_exp_net.train()

    end = time.time()

    ## FOR LOOP HERE over train_loader
    for i, (tgt_img, ref_imgs, intrinsics, _) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0
        
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        # print("tgt_img: ", tgt_img.size())

        # Forward pass
        disparities, uncertainities = disp_net(tgt_img)
        # disparities = outputs[:, :1, :, :]
        # uncertainities = outputs[:, 1:, :, :]
        depth = [1/disp for disp in disparities]
        explainability_mask, pose = pose_exp_net(tgt_img, ref_imgs)
        # embed()

        # assert((w2> 0.0) == args.uncert, "Choose between Uncertainity and Explainability")
        
        # COMPUTE LOSSES
        loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                                               depth, explainability_mask, pose, uncertainities, args.uncert_loss_epsilon, args.uncert, 
                                                               args.rotation_mode, args.padding_mode)
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if log_losses:
            tb_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            if w2 > 0:
                tb_writer.add_scalar('explanability_loss', loss_2.item(), n_iter)
            tb_writer.add_scalar('disparity_smoothness_loss', loss_3.item(), n_iter)
            tb_writer.add_scalar('total_loss', loss.item(), n_iter)

        if log_output:
            tb_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)
            for k, scaled_maps in enumerate(zip(depth, disparities, warped, diff, explainability_mask)):
                log_output_tensorboard(tb_writer, "train", 0, " {}".format(k), n_iter, *scaled_maps)

        losses.update(loss.item(), args.batch_size)

        # Gradient compute and Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        print("Training Epoch: {} Time: {}, Loss: {}".format(epoch, batch_time, loss.item()))
        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item()])
        if i >= epoch_size - 1:
            break
        n_iter += 1
    return losses.avg[0]


@torch.no_grad()    # Avoids gradient updation (backward pass is skipped during evaluation)
def validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, tb_writer, sample_nb_to_log=3):
    global device
    print("Validating without GT")
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)

    log_outputs = sample_nb_to_log > 0
    # Output the logs throughout the whole dataset
    batches_to_log = list(np.linspace(0, len(val_loader), sample_nb_to_log).astype(int))
    
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight
    poses = np.zeros(((len(val_loader)-1) * args.batch_size * (args.sequence_length-1), 6))
    disp_values = np.zeros(((len(val_loader)-1) * args.batch_size * 3))


    
    # Evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # embed()
        disp, uncert = disp_net(tgt_img)
        # disp = disp_uncert[:, :1, :, :]
        # uncert = disp_uncert[:, 1:, :, :]
        depth = 1/disp
        explainability_mask, pose = pose_exp_net(tgt_img, ref_imgs)

        loss1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs,
                                                               intrinsics, depth,
                                                               explainability_mask, pose, uncert, args.uncert_loss_epsilon, args.uncert, 
                                                               args.rotation_mode, args.padding_mode)
        if w2 > 0:
            loss2 = explainability_loss(explainability_mask).item()
        else:
            loss2 = 0
        loss3 = smooth_loss(depth).item()

        if log_outputs and i in batches_to_log:  # log first output of wanted batches
            index = batches_to_log.index(i)
            if epoch == 0:
                for j, ref in enumerate(ref_imgs):
                    tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(tgt_img[0]), 0)
                    tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(ref[0]), 1)

            log_output_tensorboard(tb_writer, 'val', index, '', epoch, 1./disp, disp, warped[0], diff[0], explainability_mask)

        if log_outputs and i < len(val_loader)-1:
            step = args.batch_size*(args.sequence_length-1)
            poses[i * step:(i+1) * step] = pose.cpu().view(-1, 6).numpy()
            step = args.batch_size * 3
            disp_unraveled = disp.cpu().view(args.batch_size, -1)
            disp_values[i * step:(i+1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                            disp_unraveled.median(-1)[0],
                                                            disp_unraveled.max(-1)[0]]).numpy()

        loss = w1*loss1 + w2*loss2 + w3*loss3
        losses.update([loss, loss1, loss2, loss3])

        print('valid: Time {} Loss {}'.format(batch_time, losses))
    tb_writer.add_histogram('disp_values', disp_values, epoch)
    return losses.avg, ['Validation Total loss', 'Validation Photo loss', 'Validation Exp loss', 'Smoothness Loss']

@torch.no_grad()
def validate_with_gt_pose(args, val_loader, disp_net, pose_exp_net, epoch, tb_writer, sample_nb_to_log=3):
    global device
    # batch_time = AverageMeter()
    # batch_time = AverageMeter()
    depth_error_names = ['abs_rel', 'sq_rel']
    depth_errors = AverageMeter(i=len(depth_error_names), precision=4)
    pose_error_names = ['ATE']
    pose_errors = AverageMeter(i=1, precision=4)
    log_outputs = sample_nb_to_log > 0

    batches_to_log = list(np.linspace(0, len(val_loader), sample_nb_to_log).astype(int))
    poses_values = np.zeros(((len(val_loader)-1) * args.batch_size * (args.sequence_length-1), 6))
    disp_values = np.zeros(((len(val_loader)-1) * args.batch_size * 3))

    disp_net.eval()
    pose_exp_net.eval()

    print("Validate with GT, pose.. val: ", len(val_loader))

    for i, (tgt_img, ref_imgs, gt_depth, gt_poses) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        gt_depth = gt_depth.to(device)
        gt_poses = gt_poses.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        b = tgt_img.shape[0]

        # Compute Output
        output_disp, output_uncert = disp_net(tgt_img)
        # output_disp = disp_uncert[:, :1, :, :]
        # output_uncert = disp_uncert[:, 1:, :, :]
        output_depth = 1/output_disp
        explainability_mask, output_poses = pose_exp_net(tgt_img, ref_imgs)

        # print("reordered output poses: ", output_poses[:, :gt_poses.shape[1]//2])
        # print("Size of output_pose: ", output_poses.size(), gt_poses.size())

        reordered_output_poses = torch.cat([output_poses[:, :gt_poses.shape[1]//2],
                                            torch.zeros(b, 1, 6).to(output_poses),
                                            output_poses[:, gt_poses.shape[1]//2:]], dim=1)

        unravelled_poses = reordered_output_poses.reshape(-1, 6)
        unravelled_matrices = pose_vec2mat(unravelled_poses, rotation_mode=args.rotation_mode)
        inv_transform_matrices = unravelled_matrices.reshape(b, -1, 3, 4)

        rot_matrices = inv_transform_matrices[..., :3].transpose(-2, -1)
        tr_vectors = -rot_matrices @ inv_transform_matrices[..., -1:]

        transform_matrices = torch.cat([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices.reshape(b, -1, 3, 4)[:, :1]
        final_poses = first_inv_transform[..., :3] @ transform_matrices
        final_poses[..., -1:] += first_inv_transform[..., -1:]
        final_poses = final_poses.reshape(b, -1, 3, 4)

        if log_outputs and i in batches_to_log:  # log first output of wanted batches
            index = batches_to_log.index(i)
            if epoch == 0:
                for j, ref in enumerate(ref_imgs):
                    tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(tgt_img[0]), 0)
                    tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(ref[0]), 1)

            log_output_tensorboard(tb_writer, 'val', index, '', epoch, output_depth, output_disp, None, None, explainability_mask)

        if log_outputs and i < len(val_loader)-1:
            step = args.batch_size*(args.sequence_length-1)
            poses_values[i * step:(i+1) * step] = output_poses.cpu().view(-1, 6).numpy()
            step = args.batch_size * 3
            disp_unraveled = output_disp.cpu().view(args.batch_size, -1)
            disp_values[i * step:(i+1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                            disp_unraveled.median(-1)[0],
                                                            disp_unraveled.max(-1)[0]]).numpy()

        depth_errors.update(compute_depth_errors(gt_depth, output_depth[:, 0]))
        pose_errors.update(compute_pose_error(gt_poses, final_poses))

        if i % args.print_freq == 0:
            print('valid: Abs Rel Error {:.4f} ({:.4f}), Sq Rel Error: {:.4f} ({:.4f}), ATE {:.4f} ({:.4f})'.format(depth_errors.val[0],
                                                                                       depth_errors.avg[0],
                                                                                       depth_errors.val[1],
                                                                                       depth_errors.avg[1],
                                                                                       pose_errors.val[0],
                                                                                       pose_errors.avg[0]))
            # logger.valid_writer.write(
            #     'valid: Time {} Abs Error {:.4f} ({:.4f}), ATE {:.4f} ({:.4f})'.format(batch_time,
            #                                                                            depth_errors.val[0],
            #                                                                            depth_errors.avg[0],
            #                                                                            pose_errors.val[0],
            #                                                                           pose_errors.avg[0]))
        
    print("Return validation output")
    return depth_errors.avg + pose_errors.avg, depth_error_names + pose_error_names



if __name__ == '__main__':
    main()
