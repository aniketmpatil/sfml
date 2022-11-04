import random
import numpy as np
from path import Path
from imageio import imread
import torch.utils.data as data

"""
    Documentation of creating Custom Dataset: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#
"""

def load_as_float(path):
    return imread(path).astype(np.float32)

class SequenceFolder():
    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        '''
            The __init__ function is run once when instantiating the Dataset object.
        '''
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)

        ## Get scenes from val.txt or train.txt
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]

        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        '''
            Gets the samples in a list, where each sample contains a target image, and a number of source images (based on sequence length)
        '''
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        '''
            The __getitem__ function loads and returns a sample from the dataset at the given index idx
        '''
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        '''
            The __len__ function returns the number of samples in our dataset.
        '''
        return len(self.samples)


class ValidationSetWithPose(data.Dataset):
    """A sequence validation data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_1/cam.txt
        root/scene_1/pose.txt
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .
    """

    def __init__(self, root, seed=None, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            poses = np.genfromtxt(scene/'poses.txt').reshape((-1, 3, 4))
            poses_4D = np.zeros((poses.shape[0], 4, 4)).astype(np.float32)
            poses_4D[:, :3] = poses
            poses_4D[:, 3, 3] = 1
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            assert(len(imgs) == poses.shape[0])
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                tgt_img = imgs[i]
                d = tgt_img.dirname()/(tgt_img.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                sample = {'intrinsics': intrinsics, 'tgt': tgt_img, 'ref_imgs': [], 'poses': [], 'depth': d}
                first_pose = poses_4D[i - demi_length]
                sample['poses'] = (np.linalg.inv(first_pose) @ poses_4D[i - demi_length: i + demi_length + 1])[:, :3]
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sample['poses'] = np.stack(sample['poses'])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        depth = np.load(sample['depth']).astype(np.float32)
        poses = sample['poses']
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, _ = self.transform([tgt_img] + ref_imgs, None)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]

        return tgt_img, ref_imgs, depth, poses

    def __len__(self):
        return len(self.samples)