'''
    Single-View Depth Prediction Model Architecture
    DispNet Architecture
    Reused parts of code from https://github.com/ClementPinard/SfmLearner-Pytorch
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from models.model_utils import upconv, conv, downsample_conv, resize_like

# Checks if cuda can be used otherwwise uses cpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from IPython import embed

def downsample_conv(in_planes, out_planes, kernel_size=3):
    '''
        Convolution layers in contracting part   
    '''
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def conv(in_planes, out_planes):
    '''
        Convolution layers in the expanding part
    '''
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def upconv(in_planes, out_planes):
    '''
        Upconvolution layers in the expanding part
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

def predict_disp_and_uncertainity(in_planes):
    '''
        Prediction Network
    '''
    return nn.Sequential(
        nn.Conv2d(in_planes, 2, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

# def crop_like(input, ref):
#     '''
#         Match W and H of input image to reference image
#     '''
#     assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
#     return input[:, :, :ref.size(2), :ref.size(3)]

class DispNetS(nn.Module):
    def __init__(self, alpha = 10, beta = 0.01):
        super(DispNetS, self).__init__()

        ## Alpha and Beta used for?? Scaling and offset of output?
        self.alpha = alpha          
        self.beta = beta

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        upconv_planes = [512, 512, 256, 128, 64, 32, 16]

        self.conv1 = downsample_conv(             3, conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        ## Concatenation based Skip Connections
        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp_and_uncertainity(upconv_planes[3])     ## Why only Sigmoid??
        self.predict_disp3 = predict_disp_and_uncertainity(upconv_planes[4])
        self.predict_disp2 = predict_disp_and_uncertainity(upconv_planes[5])
        self.predict_disp1 = predict_disp_and_uncertainity(upconv_planes[6])

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = resize_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = resize_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = resize_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = resize_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        op_layer_4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta
        disp4 = op_layer_4[:, :1, :, :]

        # print("Pred_Disp output shape:", op_layer_4.shape)
        # embed()
        out_upconv3 = resize_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = resize_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        op_layer_3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
        disp3 = op_layer_3[:, :1, :, :]

        out_upconv2 = resize_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = resize_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        op_layer_2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta
        disp2 = op_layer_2[:, :1, :, :]

        out_upconv1 = resize_like(self.upconv1(out_iconv2), x)
        disp2_up = resize_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        op_layer_1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        disp1 = op_layer_1[:, :1, :, :]

        uncert1 = op_layer_1[:, 1:, :, :]
        uncert2 = op_layer_2[:, 1:, :, :]
        uncert3 = op_layer_3[:, 1:, :, :]
        uncert4 = op_layer_4[:, 1:, :, :]
        # embed()

        if self.training:
            return [disp1, disp2, disp3, disp4], [uncert1, uncert2, uncert3, uncert4]
        else:
            return disp1, uncert1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)


# if __name__ == '__main__':
#     dispnet = DispNetS().to(device)

#     print(dispnet)
