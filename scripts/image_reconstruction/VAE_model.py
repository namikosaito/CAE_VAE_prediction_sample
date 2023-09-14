 # -*- coding: utf-8 -*-

from functools import singledispatchmethod
from numpy.core.defchararray import decode, encode
import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
class VAE_mid(nn.Module):
    def __init__(self, height, width, color_ch, cnn_ch0, cnn_ch1, cnn_ch2, cnn_ch3, h_dim0, h_dim1, h_dim2, output_dim):
        super(VAE_mid, self).__init__()
        convoluted_size = [cnn_ch3, 1, int(height/2/2/2/2), int(width/2/2/2//2)]

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=color_ch, out_channels=cnn_ch0, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch0),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch0,  out_channels=cnn_ch1, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch1),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch1,  out_channels=cnn_ch2, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch2),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch2,  out_channels=cnn_ch3, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch3),
            nn.ReLU()
        )
        self.encoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3], h_dim0),
            nn.BatchNorm1d(h_dim0),
            nn.ReLU(),
            nn.Linear(h_dim0, h_dim1),
            nn.BatchNorm1d(h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.BatchNorm1d(h_dim2),
            nn.Sigmoid(),
        )
        self.encoder_mean = nn.Sequential(
            nn.Linear(h_dim2, h_dim2)
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(h_dim2, h_dim2),
            nn.Softplus()
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(h_dim2, h_dim1),
            nn.BatchNorm1d(h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim0),
            nn.BatchNorm1d(h_dim0),
            nn.ReLU(),
            nn.Linear(h_dim0, convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3]),
            nn.BatchNorm1d(convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3]),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=cnn_ch3, out_channels=cnn_ch2,  kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch2, out_channels=cnn_ch1,  kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch1, out_channels=cnn_ch0,  kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch0, out_channels=color_ch, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.ReLU(),
        )
        
        self.middle_linear = nn.Sequential(
            nn.Linear(h_dim2, output_dim),
            nn.Sigmoid()
        )

    def reparametrizaion(self, mean, var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon*torch.exp(0.5*var)  #torch.sqrt(var)

    def encode(self, x):
        h = self.encoder_conv(x)
        self.size = h.size()
        h = self.encoder_linear(h)
        mean = self.encoder_mean(h)
        var = self.encoder_var(h)
        prediction = self.middle_linear(h)
        return mean, var, prediction

    def decode(self, h, size=None):
        if size != None:
            self.size = size
        h = self.decoder_linear(h)
        h = h.reshape(self.size)
        x = self.decoder_conv(h)
        return x
    
    def forward(self, x, device):
        mean, var, prediction = self.encode(x)
        z = self.reparametrizaion(mean, var, device)
        x_hat = self.decode(z)
        return z, x_hat, prediction, mean, var