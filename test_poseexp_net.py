import torch, argparse
import numpy as np
from tqdm import tqdm
from path import Path

from inverse_warp import pose_vec2mat
from models import PoseExpNet
from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework

parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MIN_DEPTH = 1e-3
MAX_DEPTH = 80
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 416
model_path = 'saved_models/exp_pose_model_best.pth.tar'
dataset_dir = ''

@torch.no_grad()
def main():
    args = parser.parse_args()

    weights = torch.load(model_path)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    framework = test_framework(args.dataset_dir, args.sequences, seq_length)
    ATE_list = np.zeros((len(framework), 1), np.float32)

    assert(args.output_dir != None) 
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']
        h,w,_ = imgs[0].shape

        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).to(device)
            if i == len(imgs)//2:
                tgt_img = img
            else:
                ref_imgs.append(img)

        # Forward Pass
        exp, poses = pose_net(tgt_img, ref_imgs)
        # print(poses.shape)

        poses = poses.cpu()[0]
        poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])

        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]

        if args.output_dir is not None:
            predictions_array[j] = final_poses

        ATE = compute_pose_error(sample['poses'], final_poses)
        ATE_list[j] = ATE

    mean_errors = ATE_list.mean(0)
    std_errors = ATE_list.std(0)
    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)

def compute_pose_error(gt, pred):
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))

    return ATE/snippet_length

if __name__ == '__main__':
    main()