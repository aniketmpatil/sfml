import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inverse_warp import inverse_warp

def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, depth, 
                                    explainability_mask, pose, 
                                    rotation_mode='euler', padding_mode='zeros'):
    def one_scale(depth, explainability_mask):
        assert(pose.size(1) == len(ref_imgs))
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
    
        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        # print("tgt_: ", tgt_img.size())
        # print("ref_: ", ref_imgs[0].size())
        # print("depth_: ", depth.size())

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        # print("tgt: ", tgt_img_scaled.size())
        # print("ref: ", ref_imgs_scaled[0].size())
        # print("depth: ", depth.size())

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose, 
                                        intrinsics_scaled, rotation_mode, padding_mode)
            ## valid_points is a tensor that contains bool values of size (4, 128, 416) to (4, 16, 52)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)  #???

            reconstruction_loss += diff.abs().mean()

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])
        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]
    
    total_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results

def explainability_loss(mask):
    '''
        Calculates Binary Cross Entropy Loss between a ones tensor and explainability mask
    '''
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def smooth_loss(pred_map):
    '''
        Smoothing loss, calculated using gradients in x and y direction, followed by 
    '''
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3
    return loss


