from loss_functions import smooth_loss, explainability_loss, photometric_reconstruction_loss
from logger import AverageMeter
import torch
import time
import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn

from datasets.sequence_folders import SequenceFolder
import custom_transforms as cust_trans
import models
# import models.DispNetS as DispNetS
# import models.PoseExpNet as PoseExpNet

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
# parser.add_argument('--with-gt', action='store_true', help='use depth ground truth for validation. '
#                     'You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
# parser.add_argument('--with-pose', action='store_true', help='use pose ground truth for validation. '
#                     'You need to store it in text files of 12 columns see data/kitti_raw_loader.py for an example '
#                     'Note that for kitti, it is recommend to use odometry train set to test pose')
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
# parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
#                     help='csv where to save per-epoch train and valid stats')
# parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
#                     help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
# parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
# parser.add_argument('-f', '--training-output-freq', type=int,
#                     help='frequence for outputting dispnet outputs and warped imgs at training for all scales. '
#                          'if 0, will not output',
#                     metavar='N', default=0)

if torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")

def main():
    ## SETUP
    global device
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # if args.evaluate:
    #     args.epochs = 0

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

    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )
    val_set = SequenceFolder(                   ## Check other conditions like when GT is available
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

    for epoch in range(args.epochs):
        print("Epoch :", epoch)

        train_loss = train(args, train_loader, disp_net, pose_exp_net, optimizer, args.epoch_size)

        error, error_names = validate_with_gt_pose(args, val_loader, disp_net, pose_exp_net, epoch)

def train(args, train_loader, disp_net, pose_exp_net, optimizer, epoch_size):
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight
    
    # Train mode
    disp_net.train()
    pose_exp_net.train()

    end = time.time()
    avg_loss = 0

    ## FOR OOP HERE over train_loader
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        # print("tgt_img: ", tgt_img.size())

        # Forward pass
        disparities = disp_net(tgt_img)
        depth = [1/disp for disp in disparities]
        explainability_mask, pose = pose_exp_net(tgt_img, ref_imgs)

        # COMPUTE LOSSES
        loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                                               depth, explainability_mask, pose,
                                                               args.rotation_mode, args.padding_mode)
        loss_2 = explainability_loss(explainability_mask)
        loss_3 = smooth_loss(depth)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        losses.update(loss.item(), args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        print("Train: Time: {}, Loss: {}".format(batch_time, loss.item()))

    return losses.avg[0]
        

@torch.no_grad()    # Avoids gradient updation (backward pass is skipped during evaluation)
def validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, sample_nb_to_log=3):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)


    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight


    
    # Evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

@torch.no_grad()
def validate_with_gt_pose(args, val_loader, disp_net, pose_exp_net, epoch, sample_nb_to_log=3):
    pass

if __name__ == '__main__':
    main()