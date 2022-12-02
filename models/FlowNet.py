'''
    FlowNet Architecture
    Reused parts of code from:
    1. https://github.com/yijie0710/GeoNet_pytorch
    2. https://github.com/ClementPinard/SfmLearner-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init import xavier_normal_, zeros_
from model_utils import upconv, conv, downsample_conv, resize_like

def get_flow(in_chnls):
    return nn.Conv2d(in_chnls,2,kernel_size=1,padding=0)

class FlowNet(nn.Module):
    def __init__(self, input_channels, flow_scale_factor):
        super(FlowNet, self).__init__()

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        upconv_planes = [512, 512, 256, 128, 64, 32, 16]

        self.conv1 = downsample_conv(input_channels, conv_planes[0], kernel_size=7)
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

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(2 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(2 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(2 + upconv_planes[6], upconv_planes[6])

        self.flow4 = get_flow(128)
        self.flow3 = get_flow(64)
        self.flow2 = get_flow(32)
        self.flow1 = get_flow(16)

        self.alpha = flow_scale_factor
        self.beta = 0
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
    
    def forward(self,x):
        #encode
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        # decode
        out_upconv7 = resize_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = resize_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5),1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = resize_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = resize_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3),1)
        out_iconv4 = self.iconv4(concat4)
        out_flow4 = self.alpha*self.flow4(out_iconv4)+self.beta

        out_upconv3 = resize_like(self.upconv3(out_iconv4), out_conv2)
        out_upflow4 = resize_like(F.interpolate(
            out_flow4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, out_upflow4), 1)
        out_iconv3 = self.iconv3(concat3)
        out_flow3 = self.alpha*self.flow3(out_iconv3)+self.beta

        out_upconv2 = resize_like(self.upconv2(out_iconv3), out_conv1)
        out_upflow3 = resize_like(F.interpolate(
            out_flow3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, out_upflow3), 1)
        out_iconv2 = self.iconv2(concat2)
        out_flow2 = self.alpha*self.flow2(out_iconv2)+self.beta

        out_upconv1 = resize_like(self.upconv1(out_iconv2), x)
        out_upflow2 = resize_like(F.interpolate(
            out_flow2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, out_upflow2), 1)
        out_iconv1 = self.iconv1(concat1) 
        out_flow1 = self.alpha*self.flow1(out_iconv1)+self.beta

        return out_flow1, out_flow2, out_flow3, out_flow4

        ## Following DispNet logic. Check this part again!
        # if self.training:
        #     return out_flow1, out_flow2, out_flow3, out_flow4
        # else:
        #     return out_flow1
