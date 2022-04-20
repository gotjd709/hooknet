"""
Hooknet
description: The same model as the structure in the paper. (https://arxiv.org/abs/2006.12230)
Number of input image: 2 images with the same size and center but with different resolution (Target: high resolution, Context: low resolution)
size: (2, 3, 284, 284) -> (70, 70)

Quad scale hooknet
description: My idea.
Number of input image: 4 images with the same size and center but with different resolution
size: (4, 3, 284, 284) -> (70, 70)

"""


import torch
import torch.nn as nn

### Hooknet

class Backbone_Block(nn.Module):
    
    def __init__(self, in_c, out_c, k_size, stride, padding):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(num_features=out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        
        return x

class Hooknet(nn.Module):
    
    def __init__(self, in_channels, class_num, hook_indice):
        super().__init__()

        self.hook_indice = hook_indice
        self.block11_1 = Backbone_Block(in_channels, 16, 3, 1, 0)
        self.block11_2 = Backbone_Block(16, 16, 3, 1, 0)
        self.maxp11_3 = nn.MaxPool2d(2, 2, 0)
        
        self.block21_1 = Backbone_Block(16, 32, 3, 1, 0)
        self.block21_2 = Backbone_Block(32, 32, 3, 1, 0)
        self.maxp21_3 = nn.MaxPool2d(2, 2, 0)
        
        self.block31_1 = Backbone_Block(32, 64, 3, 1, 0)
        self.block31_2 = Backbone_Block(64, 64, 3, 1, 0)
        self.maxp31_3 = nn.MaxPool2d(2, 2, 0)
        
        self.block41_1 = Backbone_Block(64, 128, 3, 1, 0)
        self.block41_2 = Backbone_Block(128, 128, 3, 1, 0)
        self.maxp41_3 = nn.MaxPool2d(2, 2, 0)            

        self.block51_1 = Backbone_Block(128, 160, 3, 1, 0)
        self.block51_2 = Backbone_Block(160, 160, 3, 1, 0) 

        self.cont12_1_1 = nn.ConvTranspose2d(160, 160, 2, 2, 0)
        if self.hook_indice[0] == 4:
            if self.hook_indice[1] == 1:
                self.cont12_1_2 = nn.ConvTranspose2d(192, 160, 2, 2, 0)
            elif self.hook_indice[1] == 2:
                self.cont12_1_2 = nn.ConvTranspose2d(224, 160, 2, 2, 0)
            elif self.hook_indice[1] == 3:
                self.cont12_1_2 = nn.ConvTranspose2d(288, 160, 2, 2, 0)
            else:
                raise ValueError(f'The hook_indice should be hook_indice[0](= {self.hook_indice[0]}) > hook_indice[1](= {self.hook_indice[1]}) > 0. Check your hook_indice.')
        else:
            self.cont12_1_2 = nn.ConvTranspose2d(160, 160, 2, 2, 0)
        self.block12_2 = Backbone_Block(160, 128, 3, 1, 0)
        self.block12_3 = Backbone_Block(256, 128, 3, 1, 0)
        self.block12_4 = Backbone_Block(128, 128, 3, 1, 0)

        self.cont22_1_1 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        if self.hook_indice[0] == 3:
            if self.hook_indice[1] == 1:
                self.cont22_1_2 = nn.ConvTranspose2d(160, 128, 2, 2, 0)
            elif self.hook_indice[1] == 2:
                self.cont22_1_2 = nn.ConvTranspose2d(192, 128, 2, 2, 0)
            else:
                raise ValueError(f'The hook_indice should be hook_indice[0](= {self.hook_indice[0]}) > hook_indice[1](= {self.hook_indice[1]}) > 0. Check your hook_indice.')
        else:
            self.cont22_1_2 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.block22_2 = Backbone_Block(128, 64, 3, 1, 0)
        self.block22_3 = Backbone_Block(128, 64, 3, 1, 0)
        self.block22_4 = Backbone_Block(64, 64, 3, 1, 0)

        self.cont32_1_1 = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        if self.hook_indice[0] == 2:
            if self.hook_indice[1] == 1:
                self.cont32_1_2 = nn.ConvTranspose2d(96, 64, 2, 2, 0)
            else:
                raise ValueError(f'The hook_indice should be hook_indice[0](= {self.hook_indice[0]}) > hook_indice[1](= {self.hook_indice[1]}) > 0. Check your hook_indice.')
        else:
            self.cont32_1_2 = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.block32_2 = Backbone_Block(64, 32, 3, 1, 0)
        self.block32_3 = Backbone_Block(64, 32, 3, 1, 0)
        self.block32_4 = Backbone_Block(32, 32, 3, 1, 0)

        self.cont42_1_1 = nn.ConvTranspose2d(32, 32, 2, 2, 0)
        self.cont42_1_2 = nn.ConvTranspose2d(32, 32, 2, 2, 0)
        self.block42_2 = Backbone_Block(32, 16, 3, 1, 0)
        self.block42_3 = Backbone_Block(32, 16, 3, 1, 0)
        self.block42_4 = Backbone_Block(16, 16, 3, 1, 0)

        
        self.output1 = nn.Conv2d(16, class_num, 3, 1, 1)
        self.output2 = nn.Softmax()
        
    def context(self, x):
        # Contracting Path
        #x = x.permute(1,0,2,3,4)
        x11_1 = self.block11_1(x[:,1,...])
        x11_2 = self.block11_2(x11_1)
        x11_3 = self.maxp11_3(x11_2)
        
        x21_1 = self.block21_1(x11_3)
        x21_2 = self.block21_2(x21_1)
        x21_3 = self.maxp21_3(x21_2)
        
        x31_1 = self.block31_1(x21_3)
        x31_2 = self.block31_2(x31_1)
        x31_3 = self.maxp31_3(x31_2)
        
        x41_1 = self.block41_1(x31_3)
        x41_2 = self.block41_2(x41_1)
        x41_3 = self.maxp41_3(x41_2)
        
        x51_1 = self.block51_1(x41_3)
        x51_2 = self.block51_2(x51_1)

        # Expanding Path
        x12_1_1 = self.cont12_1_1(x51_2)
        x12_2 = self.block12_2(x12_1_1)        
        x12_3 = self.block12_3(torch.cat([x12_2, x41_2[:, :, x41_2.shape[2]//2-9:x41_2.shape[2]//2+9, x41_2.shape[3]//2-9:x41_2.shape[3]//2+9]], dim=1))
        x12_4 = self.block12_4(x12_3)
        
        if self.hook_indice[1] == 3:
            return x12_4
        x22_1_1 = self.cont22_1_1(x12_4)
        x22_2 = self.block22_2(x22_1_1)
        x22_3 = self.block22_3(torch.cat([x22_2, x31_2[:, :, x31_2.shape[2]//2-13:x31_2.shape[2]//2+13, x31_2.shape[3]//2-13:x31_2.shape[3]//2+13]], dim=1))
        x22_4 = self.block22_4(x22_3)
        
        if self.hook_indice[1] == 2:
            return x22_4
        x32_1_1 = self.cont32_1_1(x22_4)
        x32_2 = self.block32_2(x32_1_1)
        x32_3 = self.block32_3(torch.cat([x32_2, x21_2[:, :, x21_2.shape[2]//2-21:x21_2.shape[2]//2+21, x21_2.shape[3]//2-21:x21_2.shape[3]//2+21]], dim=1))
        x32_4 = self.block32_4(x32_3)

        if self.hook_indice[1] == 1:
            return x32_4    
        x42_1_1 = self.cont42_1_1(x32_4)
        x42_2 = self.block42_2(x42_1_1)
        x42_3 = self.block42_3(torch.cat([x42_2, x11_2[:, :, x11_2.shape[2]//2-37:x11_2.shape[2]//2+37, x11_2.shape[3]//2-37:x11_2.shape[3]//2+37]], dim=1))
        x42_4 = self.block42_4(x42_3)
        
        output1 = self.output1(x42_4)
        output2 = self.output2(output1)        
        return output2
    
    def forward(self, x):
        # Contracting Path
        #x = x.permute(1,0,2,3,4)
        x11_1 = self.block11_1(x[:,0,...])
        x11_2 = self.block11_2(x11_1)
        x11_3 = self.maxp11_3(x11_2)
        
        x21_1 = self.block21_1(x11_3)
        x21_2 = self.block21_2(x21_1)
        x21_3 = self.maxp21_3(x21_2)
        
        x31_1 = self.block31_1(x21_3)
        x31_2 = self.block31_2(x31_1)
        x31_3 = self.maxp31_3(x31_2)
        
        x41_1 = self.block41_1(x31_3)
        x41_2 = self.block41_2(x41_1)
        x41_3 = self.maxp41_3(x41_2)
        
        x51_1 = self.block51_1(x41_3)
        x51_2 = self.block51_2(x51_1)

        # Expanding Path
        if self.hook_indice[0] == 4:
            y = self.context(x)
            x51_2 = torch.cat([x51_2, y[:, :, y.shape[2]//2-5:y.shape[2]//2+5, y.shape[3]//2-5:y.shape[3]//2+5]], dim=1) 
        x12_1_2 = self.cont12_1_2(x51_2)
        x12_2 = self.block12_2(x12_1_2)        
        x12_3 = self.block12_3(torch.cat([x12_2, x41_2[:, :, x41_2.shape[2]//2-9:x41_2.shape[2]//2+9, x41_2.shape[3]//2-9:x41_2.shape[3]//2+9]], dim=1))
        x12_4 = self.block12_4(x12_3)

        if self.hook_indice[0] == 3:
            y = self.context(x)
            x12_4 = torch.cat([x12_4, y[:, :, y.shape[2]//2-7:y.shape[2]//2+7, y.shape[3]//2-7:y.shape[3]//2+7]], dim=1)
        x22_1_2 = self.cont22_1_2(x12_4)
        x22_2 = self.block22_2(x22_1_2)
        x22_3 = self.block22_3(torch.cat([x22_2, x31_2[:, :, x31_2.shape[2]//2-13:x31_2.shape[2]//2+13, x31_2.shape[3]//2-13:x31_2.shape[3]//2+13]], dim=1))
        x22_4 = self.block22_4(x22_3)

        if self.hook_indice[0] == 2:
            y = self.context(x)
            x22_4 = torch.cat([x22_4, y[:, :, (y.shape[2]//2)-11:(y.shape[2]//2)+11, (y.shape[3]//2)-11:(y.shape[3]//2)+11]], dim=1)
        x32_1_2 = self.cont32_1_2(x22_4)
        x32_2 = self.block32_2(x32_1_2)
        x32_3 = self.block32_3(torch.cat([x32_2, x21_2[:, :, x21_2.shape[2]//2-21:x21_2.shape[2]//2+21, x21_2.shape[3]//2-21:x21_2.shape[3]//2+21]], dim=1))
        x32_4 = self.block32_4(x32_3)
        
        if self.hook_indice[0] == 1:
            y = self.context(x, self.hook_indice)
            x32_4 = torch.cat([x32_4, y[:, :, y.shape[2]//2-19:y.shape[2]//2+19, y.shape[3]//2-19:y.shape[3]//2+19]], dim=1)
        x42_1_2 = self.cont42_1_2(x32_4)
        x42_2 = self.block42_2(x42_1_2)
        x42_3 = self.block42_3(torch.cat([x42_2, x11_2[:, :, x11_2.shape[2]//2-37:x11_2.shape[2]//2+37, x11_2.shape[3]//2-37:x11_2.shape[3]//2+37]], dim=1))
        x42_4 = self.block42_4(x42_3)
        
        output1 = self.output1(x42_4)
        output2 = self.output2(output1)        
        return output2

def hooknet(in_channels, class_num, hook_indice):
    return Hooknet(in_channels, class_num, hook_indice)


### Quad-Scale Hooknet

class Quad_Scale_Hooknet(nn.Module):
    
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.block11_1 = Backbone_Block(in_channels, 16, 3, 1, 0)
        self.block11_2 = Backbone_Block(16, 16, 3, 1, 0)
        self.maxp11_3 = nn.MaxPool2d(2, 2, 0)
        
        self.block21_1 = Backbone_Block(16, 32, 3, 1, 0)
        self.block21_2 = Backbone_Block(32, 32, 3, 1, 0)
        self.maxp21_3 = nn.MaxPool2d(2, 2, 0)
        
        self.block31_1 = Backbone_Block(32, 64, 3, 1, 0)
        self.block31_2 = Backbone_Block(64, 64, 3, 1, 0)
        self.maxp31_3 = nn.MaxPool2d(2, 2, 0)
        
        self.block41_1 = Backbone_Block(64, 128, 3, 1, 0)
        self.block41_2 = Backbone_Block(128, 128, 3, 1, 0)
        self.maxp41_3 = nn.MaxPool2d(2, 2, 0)        

        self.block51_1 = Backbone_Block(128, 160, 3, 1, 0)
        self.block51_2 = Backbone_Block(160, 160, 3, 1, 0) 

        self.cont12_1_1 = nn.ConvTranspose2d(160, 160, 2, 2, 0)
        self.cont12_1_2 = nn.ConvTranspose2d(192, 160, 2, 2, 0)
        self.block12_2 = Backbone_Block(160, 128, 3, 1, 0)
        self.block12_3 = Backbone_Block(256, 128, 3, 1, 0)
        self.block12_4 = Backbone_Block(128, 128, 3, 1, 0)

        self.cont22_1_1 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.cont22_1_2 = nn.ConvTranspose2d(160, 128, 2, 2, 0)
        self.block22_2 = Backbone_Block(128, 64, 3, 1, 0)
        self.block22_3 = Backbone_Block(128, 64, 3, 1, 0)
        self.block22_4 = Backbone_Block(64, 64, 3, 1, 0)

        self.cont32_1_1 = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.cont32_1_2 = nn.ConvTranspose2d(96, 64, 2, 2, 0)
        self.block32_2 = Backbone_Block(64, 32, 3, 1, 0)
        self.block32_3 = Backbone_Block(64, 32, 3, 1, 0)
        self.block32_4 = Backbone_Block(32, 32, 3, 1, 0)

        self.cont42_1_1 = nn.ConvTranspose2d(32, 32, 2, 2, 0)
        self.cont42_1_2 = nn.ConvTranspose2d(32, 32, 2, 2, 0)
        self.block42_2 = Backbone_Block(32, 16, 3, 1, 0)
        self.block42_3 = Backbone_Block(32, 16, 3, 1, 0)
        self.block42_4 = Backbone_Block(16, 16, 3, 1, 0)
        
        self.output1 = nn.Conv2d(16, class_num, 3, 1, 1)
        self.output2 = nn.Softmax()
        
    ## (mpp 2.0, 4.0, 8.0) context path
    def context(self, x):
        # Contracting Path
        x11_1 = self.block11_1(x)
        x11_2 = self.block11_2(x11_1)
        x11_3 = self.maxp11_3(x11_2)
        
        x21_1 = self.block21_1(x11_3)
        x21_2 = self.block21_2(x21_1)
        x21_3 = self.maxp21_3(x21_2)
        
        x31_1 = self.block31_1(x21_3)
        x31_2 = self.block31_2(x31_1)
        x31_3 = self.maxp31_3(x31_2)
        
        x41_1 = self.block41_1(x31_3)
        x41_2 = self.block41_2(x41_1)
        x41_3 = self.maxp41_3(x41_2)
        
        x51_1 = self.block51_1(x41_3)
        x51_2 = self.block51_2(x51_1)

        # Expanding Path
        x12_1_1 = self.cont12_1_1(x51_2)
        x12_2 = self.block12_2(x12_1_1)        
        x12_3 = self.block12_3(torch.cat([x12_2, x41_2[:, :, x41_2.shape[2]//2-9:x41_2.shape[2]//2+9, x41_2.shape[3]//2-9:x41_2.shape[3]//2+9]], dim=1))
        x12_4 = self.block12_4(x12_3)
        
        x22_1_1 = self.cont22_1_1(x12_4)
        x22_2 = self.block22_2(x22_1_1)
        x22_3 = self.block22_3(torch.cat([x22_2, x31_2[:, :, x31_2.shape[2]//2-13:x31_2.shape[2]//2+13, x31_2.shape[3]//2-13:x31_2.shape[3]//2+13]], dim=1))
        x22_4 = self.block22_4(x22_3)
        
        x32_1_1 = self.cont32_1_1(x22_4)
        x32_2 = self.block32_2(x32_1_1)
        x32_3 = self.block32_3(torch.cat([x32_2, x21_2[:, :, x21_2.shape[2]//2-21:x21_2.shape[2]//2+21, x21_2.shape[3]//2-21:x21_2.shape[3]//2+21]], dim=1))
        x32_4 = self.block32_4(x32_3)
        return x32_4

    
    ## (mpp 1.0) target path
    def forward(self, x):
        # Contracting Path
        #x = x.permute(1,0,2,3,4)
        x11_1 = self.block11_1(x[:,0,...])
        x11_2 = self.block11_2(x11_1)
        x11_3 = self.maxp11_3(x11_2)
        
        x21_1 = self.block21_1(x11_3)
        x21_2 = self.block21_2(x21_1)
        x21_3 = self.maxp21_3(x21_2)
        
        x31_1 = self.block31_1(x21_3)
        x31_2 = self.block31_2(x31_1)
        x31_3 = self.maxp31_3(x31_2)
        
        x41_1 = self.block41_1(x31_3)
        x41_2 = self.block41_2(x41_1)
        x41_3 = self.maxp41_3(x41_2)
        
        x51_1 = self.block51_1(x41_3)
        x51_2 = self.block51_2(x51_1)

        # Expanding Path
        mpp8 = self.context(x[:,3,...])
        x12_1_2 = self.cont12_1_2(torch.cat([x51_2, mpp8[:, :, (mpp8.shape[2]//2)-5:(mpp8.shape[2]//2)+5, (mpp8.shape[3]//2)-5:(mpp8.shape[3]//2)+5]], dim=1))
        x12_2 = self.block12_2(x12_1_2)        
        x12_3 = self.block12_3(torch.cat([x12_2, x41_2[:, :, x41_2.shape[2]//2-9:x41_2.shape[2]//2+9, x41_2.shape[3]//2-9:x41_2.shape[3]//2+9]], dim=1))
        x12_4 = self.block12_4(x12_3)

        mpp4 = self.context(x[:,2,...])
        x22_1_2 = self.cont22_1_2(torch.cat([x12_4, mpp4[:, :, (mpp4.shape[2]//2)-7:(mpp4.shape[2]//2)+7, (mpp4.shape[3]//2)-7:(mpp4.shape[3]//2)+7]], dim=1))
        x22_2 = self.block22_2(x22_1_2)
        x22_3 = self.block22_3(torch.cat([x22_2, x31_2[:, :, x31_2.shape[2]//2-13:x31_2.shape[2]//2+13, x31_2.shape[3]//2-13:x31_2.shape[3]//2+13]], dim=1))
        x22_4 = self.block22_4(x22_3)

        mpp2 = self.context(x[:,1,...])
        x32_1_2 = self.cont32_1_2(torch.cat([x22_4, mpp2[:, :, (mpp2.shape[2]//2)-11:(mpp2.shape[2]//2)+11, (mpp2.shape[3]//2)-11:(mpp2.shape[3]//2)+11]], dim=1))
        x32_2 = self.block32_2(x32_1_2)
        x32_3 = self.block32_3(torch.cat([x32_2, x21_2[:, :, x21_2.shape[2]//2-21:x21_2.shape[2]//2+21, x21_2.shape[3]//2-21:x21_2.shape[3]//2+21]], dim=1))
        x32_4 = self.block32_4(x32_3)       

        x42_1_2 = self.cont42_1_2(x32_4)
        x42_2 = self.block42_2(x42_1_2)
        x42_3 = self.block42_3(torch.cat([x42_2, x11_2[:, :, x11_2.shape[2]//2-37:x11_2.shape[2]//2+37, x11_2.shape[3]//2-37:x11_2.shape[3]//2+37]], dim=1))
        x42_4 = self.block42_4(x42_3)
        
        output1 = self.output1(x42_4)
        output2 = self.output2(output1)        
        return output2

def quad_scale_hooknet(in_channels, class_num):
    return Quad_Scale_Hooknet(in_channels, class_num)