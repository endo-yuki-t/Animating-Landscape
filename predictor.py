# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class ConditionalMotionNet(torch.nn.Module):
    def __init__(self, nz=8, nout=2, beta=1./64.):
        super(ConditionalMotionNet, self).__init__()
        c_num = 128
         
        # Downsampling layers
        self.conv1 = ConvLayer(3+nz, c_num, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(c_num+nz, c_num*2, kernel_size=3, stride=2)
        self.in2 = InstanceNormalization(c_num*2)
        self.conv3 = ConvLayer(c_num*2+nz, c_num*4, kernel_size=3, stride=2)
        self.in3 = InstanceNormalization(c_num*4)
 
        # Residual layers
        self.res1 = ResidualBlock(c_num*4)
        self.res2 = ResidualBlock(c_num*4)
        self.res3 = ResidualBlock(c_num*4)
        self.res4 = ResidualBlock(c_num*4)
        self.res5 = ResidualBlock(c_num*4)
 
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(c_num*4*2, c_num*2, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNormalization(c_num*2)
        self.deconv2 = UpsampleConvLayer(c_num*2*2, c_num, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNormalization(c_num)
        self.deconv3 = UpsampleConvLayer(c_num*2, nout, kernel_size=5, stride=1, upsample=2)

        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.beta = beta

    def forward(self, x, z, frame_size=0):
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat((x, z2D), 1)
        h1 = self.relu(self.conv1(x_with_z))
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), h1.size(2), h1.size(3))
        h1_with_z =  torch.cat((h1, z2D), 1)
        h2 = self.relu(self.in2(self.conv2(h1_with_z)))
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), h2.size(2), h2.size(3))
        h2_with_z =  torch.cat((h2, z2D), 1)
        h3 = self.relu(self.in3(self.conv3(h2_with_z)))
          
        h4 = self.res1(h3)
        h4 = self.res2(h4)
        h4 = self.res3(h4)
        h4 = self.res4(h4)
        h4 = self.res5(h4)

        h4 = torch.cat((h4,h3),1)
        h5 = self.relu(self.in4(self.deconv1(h4)))
        h5 = torch.cat((h5,h2),1)
        h6 = self.relu(self.in5(self.deconv2(h5)))
        h6 = torch.cat((h6,h1),1)
        h7 = self.deconv3(h6)
         
        y = F.tanh(h7)*self.beta
        
        return y

class ConditionalAppearanceNet(torch.nn.Module):
    def __init__(self, nz=8):
        super(ConditionalAppearanceNet, self).__init__()
        c_num = 128
          
        # Downsampling layers
        self.conv1 = ConvLayer(3+nz, c_num, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(c_num+nz, c_num*2, kernel_size=3, stride=2)
        self.in2 = InstanceNormalization(c_num*2)
        self.conv3 = ConvLayer(c_num*2+nz, c_num*4, kernel_size=3, stride=2)
        self.in3 = InstanceNormalization(c_num*4)
  
        # Residual layers
        self.res1 = ResidualBlock(c_num*4)
        self.res2 = ResidualBlock(c_num*4)
        self.res3 = ResidualBlock(c_num*4)
        self.res4 = ResidualBlock(c_num*4)
        self.res5 = ResidualBlock(c_num*4)
  
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(c_num*4*2, c_num*2, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNormalization(c_num*2)
        self.deconv2 = UpsampleConvLayer(c_num*2*2, c_num, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNormalization(c_num)
        self.deconv3 = UpsampleConvLayer(c_num*2, 6, kernel_size=5, stride=1, upsample=2)

        self.fc1 = nn.Linear(c_num*4, 6)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
  
    def forward(self, x, z):
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat((x, z2D), 1)
        h1 = self.relu(self.conv1(x_with_z))
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), h1.size(2), h1.size(3))
        h1_with_z =  torch.cat((h1, z2D), 1)
        h2 = self.relu(self.in2(self.conv2(h1_with_z)))
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), h2.size(2), h2.size(3))
        h2_with_z =  torch.cat((h2, z2D), 1)
        h3 = self.relu(self.in3(self.conv3(h2_with_z)))
          
        h4 = self.res1(h3)
        h4 = self.res2(h4)
        h4 = self.res3(h4)
        h4 = self.res4(h4)
        h4 = self.res5(h4)
       
        h4 = torch.cat((h4,h3),1)
        h5 = self.relu(self.in4(self.deconv1(h4)))
        h5 = torch.cat((h5,h2),1)
        h6 = self.relu(self.in5(self.deconv2(h5)))
        h6 = torch.cat((h6,h1),1)
        h7 = self.deconv3(h6)
        al, bl = h7.split(3,dim=1)

        Y = al*x+bl
        Y = F.tanh(Y)
        return Y, al, bl
    
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = InstanceNormalization(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = InstanceNormalization(channels)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1,inplace=True)
     
    def forward(self, x, x_o=None):
        if x_o is None:
            residual = x
            out = self.conv1(self.relu(self.in1(x)))
            out = self.conv2(self.relu(self.in2(out)))
            out = out + residual
            return out
        
        residual = x
        residual_o = x_o
        out = self.conv1(self.relu(self.in1(x, x_o)))
        out_o = self.conv1(self.relu(self.in1(x_o, x_o)))
        out = self.conv2(self.relu(self.in2(out, out_o)))
        out_o = self.conv2(self.relu(self.in2(out_o, out_o)))
        out = out + residual
        out_o = out_o + residual_o
        return out, out_o


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class InstanceNormalization(torch.nn.Module):
    #Original code from https://github.com/abhiskk/fast-neural-style
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x, x_o=None):
        if x_o is None:
            x_o = x
        n = x_o.size(2) * x_o.size(3)
        t = x_o.view(x_o.size(0), x_o.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
    
class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = [
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)
    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x