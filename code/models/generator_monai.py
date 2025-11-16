from monai.networks.nets import UNet
from torchsummary import summary
import torch.nn as nn
import torch


def generator_unet(channels, strides, type_norm, num_res_units):

    # 5 layer network with down/upsampling by a factor of 2 at each layer (with 2-convolution residual units)

    unet = UNet(
        spatial_dims=2,  # 2D unet
        in_channels=1,  # grayscale images
        out_channels=1,  # grayscale images
        channels=tuple(
            channels
        ),  # number of channels in each layer - (64, 128, 256, 512, 512, 512, 512)
        strides=tuple(strides),  # for downsampling layers - (2, 2, 2, 2, 2, 2)
        kernel_size=3,
        act="RELU",
        #act = ('leakyrelu', {'inplace': True, 'negative_slope': 0.2}), # activation function
        norm=type_norm,  # normalization: "INSTANCE" or Norm.BATCH
        num_res_units=num_res_units,  # residual units per layer - 2 Conv Layer, ReLU layer -2
    )

    generator = nn.Sequential(
        unet,
        nn.Tanh(),  # clip between -1 and 1
    )

    return generator


'''
channels = "64, 128, 256, 512, 512, 512, 512, 512"
channels = list(map(int, channels.split(",")))
strides = "2,2,2,2,2,2,2"
strides = list(map(int, strides.split(",")))
type_norm = "INSTANCE"
num_res_units = 0
device = torch.device("cpu")
generator_simple = generator_unet(channels, strides, type_norm, num_res_units).to(device)
summary(generator_simple, (1,256,256), device="cpu") # without the batch
print(generator_simple)

output=generator_simple(torch.randn(1,1,256,256))
print(torch.unique(output))
'''