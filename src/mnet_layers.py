from torch import nn
import torch

class MNetUp(nn.Module):
    def __init__(self, scale_factor):
        super(MNetUp, self).__init__()
        self.scale_factor = scale_factor
        if self.scale_factor: 
            self.up = nn.Upsample(scale_factor=self.scale_factor[0])
        else:
            self.up = nn.Identity()
        
    def forward(self, x):
        return self.up(x)
    
class MNetDown(nn.Module):
    def __init__(self, kernel_size):
        super(MNetDown, self).__init__()
        self.kernel_size = kernel_size
        if self.kernel_size:
            self.down = nn.AvgPool2d(kernel_size=self.kernel_size[0])
        else:
            self.down = nn.Identity()
            
    def forward(self, x):
        return self.down(x) 
    
class MNetMaxpool(nn.Module):
    def __init__(self, kernel_size):
        super(MNetMaxpool, self).__init__()
        self.kernel_size = kernel_size
        if self.kernel_size:
            self.maxpool = nn.MaxPool2d(kernel_size=self.kernel_size[0])
        else:
            self.maxpool == nn.Identity()
    
    def forward(self, x):
        return self.maxpool(x)

class MNetDeconv(nn.Module):
    def __init__(self, deconv):
        super(MNetDeconv, self).__init__()
        self.deconv = deconv
        if self.deconv:
            self.deconvolution = nn.ConvTranspose2d(in_channels=self.deconv[0], out_channels=self.deconv[1], kernel_size=2, stride=2)
        else: 
            self.deconvolution = nn.Identity()
 
    def forward(self, x):
        return self.deconvolution(x)

class MNetConvSig(nn.Module):
    def __init__(self, conv_sig):
        super(MNetConvSig, self).__init__()
        self.conv_sig = conv_sig
        if self.conv_sig:
            self.convolution = nn.Conv2d(in_channels=self.conv_sig[0], out_channels=self.conv_sig[1], kernel_size=1, stride=1)
            self.activation = nn.Sigmoid()
            self.seq = nn.Sequential(*[self.convolution, self.activation])
        else:
            self.seq = nn.Identity()

    def forward(self, x):
        return self.seq(x)

class MNetConvRelu(nn.Module):
    def __init__(self, conv_relu):
        super(MNetConvRelu, self).__init__()
        self.conv_relu = conv_relu
        if self.conv_relu:
            self.convolution = nn.Conv2d(in_channels=self.conv_relu[0], out_channels=self.conv_relu[1], kernel_size=3, padding=1)
            self.activation = nn.ReLU()
            self.seq = nn.Sequential(*[self.convolution, self.activation])
        else:
            self.seq = nn.Identity()
            
    def forward(self, x):
        return self.seq(x)
            
class MNetConcat(nn.Module):
    def __init__(self, dim):
        super(MNetConcat, self).__init__()
        self.dim = dim
        
    def forward(self, x, y=None):
        if self.dim:
            return torch.cat((x, y), dim=self.dim[0])
        else:
            return x
    
        