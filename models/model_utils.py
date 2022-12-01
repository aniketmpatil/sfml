import torch.nn as nn

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

def upconv(in_planes, out_planes):
    '''
        Upconvolution layers in the expanding part
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
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

def resize_like(input, ref):
    '''
        Match W and H of input image to reference image
    '''
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]