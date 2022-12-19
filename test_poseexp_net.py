import torch
import numpy as np
from tqdm import tqdm
from path import Path

from inverse_warp import pose_vec2mat
from models import PoseExpNet
from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MIN_DEPTH = 1e-3
MAX_DEPTH = 80
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 416
model_path = 'saved_models/exp_pose_model_best.pth.tar'
dataset_dir = ''
sequences = ['09']
rotation_mode = 'euler' # 'quat'
output_dir_path = ""

@torch.no_grad()
def main():
    # args = parser.parse_args()

    weights = torch.load(model_path)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    framework = test_framework(dataset_dir, sequences, seq_length)
    ATE_list = np.zeros((len(framework), 1), np.float32)

    # assert(output_dir != None) 
    # output_dir = Path(output_dir_path)
    # output_dir.makedirs_p()
    # predictions_array = np.zeros((len(framework), seq_length, 3, 4))

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

        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=rotation_mode).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]

        ATE = compute_pose_error(sample['poses'], final_poses)
        ATE_list[j] = ATE

    mean_errors = ATE_list.mean(0)
    std_errors = ATE_list.std(0)
    error_names = ['ATE']
    print('')
    print("ATE Results")
    print("\t {:>10}".format(*error_names))
    print("mean \t {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}".format(*std_errors))

def compute_pose_error(gt, pred):
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))

    return ATE/snippet_length

if __name__ == '__main__':
    main()