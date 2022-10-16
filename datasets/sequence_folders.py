import random
import numpy as np
from path import Path
from imageio import imread

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
        