import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module): 
    def __init__(self, n_class=1):
        super().__init__()

        # ====================Encoder====================

        ## Block 1
        self.conve11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)               # output: 512x512x64
        self.conve12 = nn.Conv2d(64,64, kernel_size=3, padding=1)               # output: 512x512x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                      # output: 256x256x64

        ## Block 2
        self.conve21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)             # output: 256x256x128
        self.conve22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)            # output: 256x256x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                      # output: 128x128x128

        ## Block 3
        self.conve31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)            # output: 128x128x256
        self.conve32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)            # output: 128x128x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                      # output: 64x64x256

        ## Block 4
        self.conve41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)            # output: 64x64x512
        self.conve42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)            # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)                      # output: 32x32x512


        # ====================Bottleneck====================
        self.convb1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)            # output: 32x32x1024
        self.convb2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)            # output: 32x32x512


        # ====================Decoder====================

        ## Block 4
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)    # 64x64x512
        self.convd41 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)           # output 64x64x512
        self.convd42 = nn.Conv2d(512, 256, kernel_size=3, padding=1)            # output 64x64x256

        ## Block 3
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)    # 128x128x256
        self.convd31 = nn.Conv2d(512, 256, kernel_size=3, padding=1)            # output 128x128x256
        self.convd32 = nn.Conv2d(256, 128, kernel_size=3, padding=1)            # output 128x128x128

        ## Block 2 
        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)    # 256x256x128
        self.convd21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)            # output 256x256x128
        self.convd22 = nn.Conv2d(128, 64, kernel_size=3, padding=1)             # output 256x256x64

        ## Block 1 
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)      # 512x512x64
        self.convd11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)             #output 512x512x64
        self.convd12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)              #output 512x512x64

        # ====================Output Layer====================
        
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)                    # 512x512xn_class


    def forward(self, x): 
        # ====================Encoder====================

        # Block 1
        e1 = F.relu(self.conve11(x))
        e1 = F.relu(self.conve12(e1))
        p1 = self.pool1(e1)

        # Block 2
        e2 = F.relu(self.conve21(p1))
        e2 = F.relu(self.conve22(e2))
        p2 = self.pool2(e2)

        # Block 3
        e3 = F.relu(self.conve31(p2))
        e3 = F.relu(self.conve32(e3))
        p3 = self.pool3(e3)

        # Block 4
        e4 = F.relu(self.conve41(p3))
        e4 = F.relu(self.conve42(e4))
        p4 = self.pool4(e4)


        # ====================Bottleneck====================
        x = F.relu(self.convb1(p4))
        x = F.relu(self.convb2(x))


        # ====================Decoder====================

        # Block 4
        x = self.upconv4(x)
        x = F.relu(self.convd41(torch.cat([x, e4], dim=1)))
        x = F.relu(self.convd42(x))

        # Block 3
        x = self.upconv3(x)
        x = F.relu(self.convd31(torch.cat([x, e3], dim=1)))
        x = F.relu(self.convd32(x))

        # Block 2
        x = self.upconv2(x)
        x = F.relu(self.convd21(torch.cat([x, e2], dim=1)))
        x = F.relu(self.convd22(x))
        
        # Block 1
        x = self.upconv1(x)
        x = F.relu(self.convd11(torch.cat([x, e1], dim=1)))
        x = F.relu(self.convd12(x))

        return self.outconv(x)