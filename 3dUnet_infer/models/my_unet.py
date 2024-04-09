import torch
from torch import nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):  # 两次卷积封装
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),   # kernel size = 3
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),   # inplace=true覆盖原变量，节省内存
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


    def forward(self, t):
        return self.conv(t)


class ResUnet_light(nn.Module):
    def __init__(self, in_ch, out_ch, num_losses=2):
        super(ResUnet_light, self).__init__()

        self.uncertainty_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0 / num_losses], dtype=torch.float32), requires_grad=True)
            for _ in range(num_losses)
        ])
        
        # down sampling
        self.conv1 = DoubleConv(in_ch, 16)
        self.w1 = nn.Conv3d(in_ch, 16, kernel_size=1, padding=0, stride=1)
        self.pool1 = nn.MaxPool3d(2)  

        self.conv2 = DoubleConv(16, 32)  
        self.w2 = nn.Conv3d(16, 32, kernel_size=1, padding=0, stride=1)
        self.pool2 = nn.MaxPool3d(2) 

        self.conv3 = DoubleConv(32, 64)
        self.w3 = nn.Conv3d(32, 64, kernel_size=1, padding=0, stride=1)
        self.pool3 = nn.MaxPool3d(2) 

        self.conv4 = DoubleConv(64, 128)
        self.w4 = nn.Conv3d(64, 128, kernel_size=1, padding=0, stride=1)
        self.pool4 = nn.MaxPool3d(2) 

        self.conv5 = DoubleConv(128, 256)
        self.w5 = nn.Conv3d(128, 256, kernel_size=1, padding=0, stride=1)

        # up sampling0
        self.up6 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv6 = DoubleConv(256, 128)   

        self.up7 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv7 = DoubleConv(128, 64)

        self.up8 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv8 = DoubleConv(64, 32)

        self.up9 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.conv9 = DoubleConv(32, 16)

        self.conv10 = nn.Conv3d(16, 1, 1)
        
        # up sampling1
        self.up61 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv61 = DoubleConv(256, 128)   

        self.up71 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv71 = DoubleConv(128, 64)

        self.up81 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv81 = DoubleConv(64, 32)

        self.up91 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.conv91 = DoubleConv(32, 16)

        self.conv101 = nn.Conv3d(16, 1, 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.3)

    def down1(self, x):
        down0_res = self.w1(x)  
        down0 = self.conv1(x) + down0_res
        return down0

    def down1_pool(self, down0, down0_ext=0):
        down0 = down0 + down0_ext  
        down1 = self.pool1(down0)
        return down1

    def down2(self, down1):
        down1_res = self.w2(down1)
        down1 = self.conv2(down1) + down1_res
        return down1

    def down2_pool(self, down1, down1_ext=0):
        down1 = down1 + down1_ext
        down2 = self.pool2(down1)
        return down2

    def down3(self, down2):
        down2_res = self.w3(down2)
        down2 = self.conv3(down2) + down2_res
        return down2

    def down3_pool(self, down2, down2_ext=0):
        down2 = down2 + down2_ext
        down3 = self.pool3(down2)
        return down3

    def down4(self, down3, down3_ext=0):
        down3_res = self.w4(down3)
        down3 = self.conv4(down3) + down3_res + down3_ext
        return down3

    def upsample0(self, down3, down2, down1, down0):
        down4 = self.pool4(down3)
        down4_res = self.w5(down4)
        down5 = self.conv5(down4) + down4_res

        up_6 = self.up6(down5)   
        merge6 = torch.cat([up_6, down3], dim=1)    
        c6 = self.conv6(merge6)   

        up_7 = self.up7(c6)   
        merge7 = torch.cat([up_7, down2], dim=1)
        c7 = self.conv7(merge7) 

        up_8 = self.up8(c7)   
        merge8 = torch.cat([up_8, down1], dim=1)
        c8 = self.conv8(merge8) 

        up_9 = self.up9(c8)  
        merge9 = torch.cat([up_9, down0], dim=1)
        c9 = self.conv9(merge9)  

        c10 = self.conv10(c9)  

        output0 = self.sigmoid(c10)

        return output0
    
    def upsample1(self, down3, down2, down1, down0):
        down4 = self.pool4(down3)
        down4_res = self.w5(down4)
        down5 = self.conv5(down4) + down4_res

        up_6 = self.up61(down5)   
        merge6 = torch.cat([up_6, down3], dim=1)    
        c6 = self.conv61(merge6)   

        up_7 = self.up71(c6)   
        merge7 = torch.cat([up_7, down2], dim=1)
        c7 = self.conv71(merge7) 

        up_8 = self.up81(c7)   
        merge8 = torch.cat([up_8, down1], dim=1)
        c8 = self.conv81(merge8) 

        up_9 = self.up91(c8)  
        merge9 = torch.cat([up_9, down0], dim=1)
        c9 = self.conv91(merge9)  

        c10 = self.conv101(c9)  

        output1 = self.sigmoid(c10)

        return output1
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def weighted_loss(self, loss_list):
        final_loss = []
        for i, loss in enumerate(loss_list):
            weight = self.uncertainty_weights[i]
            weighted_loss = loss / (2 * weight**2) + torch.log(1 + weight)
            final_loss.append(weighted_loss)
        return sum(final_loss)