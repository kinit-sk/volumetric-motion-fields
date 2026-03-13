import torch
import torch.nn as nn
import torch.nn.functional as F


class RainNet3DNoNorm(nn.Module):
    
    def __init__(self, 
                 kernel_size = 3,
                 mode = "regression",
                 im_shape = (1, 18, 512, 512),
                 conv_shape = [["1", [4,64]],
                        ["2" , [64,128]],
                        ["3" , [128,256]],
                        ["4" , [256,512]],
                        ["5" , [512,1024]],
                        ["6" , [1536,512]],
                        ["7" , [768,256]],
                        ["8" , [384,128]],
                        ["9" , [192,64]]],
                 downsample_first_dim = False):
        
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode
        self.im_shape = im_shape

        self.conv = nn.ModuleDict()
        for name, (in_ch, out_ch) in conv_shape:
            self.conv[name] = self.make_conv_block(in_ch, out_ch ,self.kernel_size)
        
        
        if self.mode != "motion_field_volumetric":
            self.pool_final = nn.MaxPool3d(kernel_size = (2,1,1))
        else:
            self.pool_final = nn.Identity()

        if downsample_first_dim:
            self.pool = nn.MaxPool3d(kernel_size = (2,2,2))
            self.upsample = nn.Upsample(scale_factor=(2,2,2))
        else:
            self.pool = nn.MaxPool3d(kernel_size = (1,2,2))
            self.upsample = nn.Upsample(scale_factor=(1,2,2))
        
        self.drop = nn.Dropout3d(p=0.1)
        
        if self.mode == "regression":
            self.last_layer = nn.Conv3d(conv_shape[-1][-1][-1], self.im_shape[1], kernel_size=1, padding = 'valid')
        elif self.mode == "segmentation":
            self.last_layer = nn.Sequential(
                nn.Conv3d(2, 1, kernel_size=1, padding = 'valid'),
                nn.Sigmoid())
        elif self.mode == "motion_field":
            self.last_layer = nn.Sequential(
                nn.Conv3d(conv_shape[-1][-1][-1], 2, kernel_size=1, padding = 'same'))
        elif self.mode == "motion_field_channels":
            self.last_layer = nn.Sequential(
                nn.Conv3d(conv_shape[-1][-1][-1], 2*conv_shape[0][1][0], kernel_size=1, padding = 'same'))
        elif self.mode == "motion_field_volumetric":
            self.last_layer = nn.Sequential(
                nn.Conv3d(conv_shape[-1][-1][-1], 3, kernel_size=1, padding = 'same'))
        else:
            raise NotImplementedError()
            
    def make_conv_block(self, in_ch, out_ch, kernel_size):
        
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding='same', padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding='same', padding_mode='replicate'),
            nn.ReLU()
            )
    
        
    def forward(self, x):
        x1s = self.drop(self.conv["1"](x.float())) # conv1s
        x2s = self.drop(self.conv["2"](self.pool(x1s))) # conv2s
        x3s = self.drop(self.conv["3"](self.pool(x2s))) # conv3s
        x4s = self.drop(self.conv["4"](self.pool(x3s))) # conv4s
        x = self.drop(self.conv["5"](self.pool(x4s))) # conv5s
        x = torch.cat((self.upsample(x), x4s), dim=1) # up6
        x = torch.cat((self.upsample(self.drop(self.conv["6"](x))), x3s), dim=1) # up7
        x = torch.cat((self.upsample(self.drop(self.conv["7"](x))), x2s), dim=1) # up8
        x = self.pool_final(torch.cat((self.upsample(self.drop(self.conv["8"](x))), x1s), dim=1)) # up9
        x = self.pool_final(self.drop(self.conv["9"](x))) #conv9
        x = self.pool_final(self.drop(self.last_layer(x))) #outputs
        
        return x