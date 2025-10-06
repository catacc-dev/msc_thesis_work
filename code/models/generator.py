import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x
    
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1): # 4,1,256,256
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)        # 4, 64, 128, 128
        self.down2 = UNetDown(64, 128)                                 # 4, 128, 64, 64
        self.down3 = UNetDown(128, 256)                                # 4, 256, 32, 32
        self.down4 = UNetDown(256, 512, dropout=0.5)                   # 4, 512, 16, 16
        self.down5 = UNetDown(512, 512, dropout=0.5)                   # 4, 512, 8, 8
        self.down6 = UNetDown(512, 512, dropout=0.5)                   # 4, 512, 4, 4
        self.down7 = UNetDown(512, 512, dropout=0.5)                   # 4, 512, 2, 2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)  # 4, 512, 1, 1

        self.up1 = UNetUp(512, 512, dropout=0.5)                       # 4, 512, 2, 2 (concatenate with d7 -> 4, 1024, 2, 2)
        self.up2 = UNetUp(1024, 512, dropout=0.5)                      # 4, 512, 4, 4 (concatenate with d6 -> 4, 1024, 4, 4)
        self.up3 = UNetUp(1024, 512, dropout=0.5)                      # 4, 512, 8, 8 (concatenate with d5 -> 4, 1024, 8, 8)
        self.up4 = UNetUp(1024, 512, dropout=0.5)                      # 4, 512, 16, 16 (concatenate with d4 -> 4, 1024, 16, 16)
        self.up5 = UNetUp(1024, 256)                                   # 4, 256, 32, 32 (concatenate with d3 -> 4, 512, 32, 32)
        self.up6 = UNetUp(512, 128)                                    # 4, 128, 64, 64 (concatenate with d2 -> 4, 256, 64, 64)
        self.up7 = UNetUp(256, 64)                                     # 4, 64, 128, 128 (concatenate with d1 -> 4, 128, 128, 128)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),                               # 4, 128, 256, 256
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),    # 4, 1, 256, 256
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
