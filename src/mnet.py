import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # concat [pool_1_out, conv_2_1_out]: 32 + 64 = 96
        self.conv_2_2 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv_2_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3_1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        # concat [pool_2_out, conv_3_1_out]: 64 + 128 = 192
        self.conv_3_2 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.conv_3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4_1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        # concat [pool_3_out, conv_4_1_out]: 128 + 256 = 384
        self.conv_4_2 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv_4_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv1 = F.relu(self.conv_1_1(x))
        conv1 = F.relu(self.conv_1_2(conv1))
        pool1 = self.pool_1(conv1)

        scale_img_2 = F.avg_pool2d(x, kernel_size=2)
        conv2 = F.relu(self.conv_2_1(scale_img_2))
        conv2 = torch.cat([conv2, pool1], dim=1)
        conv2 = F.relu(self.conv_2_2(conv2))
        conv2 = F.relu(self.conv_2_3(conv2))
        pool2 = self.pool_2(conv2)

        scale_img_3 = F.avg_pool2d(scale_img_2, kernel_size=2)
        conv3 = F.relu(self.conv_3_1(scale_img_3))
        conv3 = torch.cat([conv3, pool2], dim=1)
        conv3 = F.relu(self.conv_3_2(conv3))
        conv3 = F.relu(self.conv_3_3(conv3))
        pool3 = self.pool_3(conv3)

        scale_img_4 = F.avg_pool2d(scale_img_3, kernel_size=2)
        conv4 = F.relu(self.conv_4_1(scale_img_4))
        conv4 = torch.cat([conv4, pool3], dim=1)
        conv4 = F.relu(self.conv_4_2(conv4))
        conv4 = F.relu(self.conv_4_3(conv4))
        pool4 = self.pool_4(conv4)

        return conv1, conv2, conv3, conv4, pool4

class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.conv_5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, pool4):
        conv5 = F.relu(self.conv_5_1(pool4))
        conv5 = F.relu(self.conv_5_2(conv5))
        return conv5

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # concat [upconv_1_out, conv_4_3_out]: 256 + 256 = 512
        self.conv_6_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # concat [upconv_2_out, conv_3_3_out]: 128 + 128 = 256
        self.conv_7_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_7_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # concat [upconv_2_out, conv_2_3_out]: 64 + 64 = 128
        self.conv_8_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_8_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv_4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # concat [upconv_2_out, conv_1_2_out]: 32 + 32 = 96
        self.conv_9_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_9_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, conv5, conv4, conv3, conv2, conv1):
        upconv6 = torch.cat([self.upconv_1(conv5), conv4], dim=1)
        conv6 = F.relu(self.conv_6_1(upconv6))
        conv6 = F.relu(self.conv_6_2(conv6))

        upconv7 = torch.cat([self.upconv_2(conv6), conv3], dim=1)
        conv7 = F.relu(self.conv_7_1(upconv7))
        conv7 = F.relu(self.conv_7_2(conv7))

        upconv8 = torch.cat([self.upconv_3(conv7), conv2], dim=1)
        conv8 = F.relu(self.conv_8_1(upconv8))
        conv8 = F.relu(self.conv_8_2(conv8))

        upconv9 = torch.cat([self.upconv_4(conv8), conv1], dim=1)
        conv9 = F.relu(self.conv_9_1(upconv9))
        conv9 = F.relu(self.conv_9_2(conv9))

        return conv6, conv7, conv8, conv9
    
class SideOutput(nn.Module):
    def __init__(self):
        super(SideOutput, self).__init__()
        self.up_1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.up_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv_out_1 = nn.Conv2d(256, 2, kernel_size=1)
        self.conv_out_2 = nn.Conv2d(128, 2, kernel_size=1)
        self.conv_out_3 = nn.Conv2d(64, 2, kernel_size=1)
        self.conv_out_4 = nn.Conv2d(32, 2, kernel_size=1)

        # self.avgpool = nn.AvgPool2d(2, stride=1)

    def forward(self, conv6, conv7, conv8, conv9):
        up1 = self.up_1(conv6)
        up2 = self.up_2(conv7)
        up3 = self.up_3(conv8)

        convout1 = F.sigmoid(self.conv_out_1(up1))
        convout2 = F.sigmoid(self.conv_out_2(up2))
        convout3 = F.sigmoid(self.conv_out_3(up3))
        convout4 = F.sigmoid(self.conv_out_4(conv9))

        # finalout = self.avgpool(torch.cat([convout1, convout2, convout3, convout4], dim=1))
        finalout = torch.mean(torch.stack([convout1, convout2, convout3, convout4], dim=1), dim=1)

        return convout1, convout2, convout3, convout4, finalout
    
class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()

        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()
        self.side_output = SideOutput()

    def forward(self, x):
        # Encoder
        conv1, conv2, conv3, conv4, pool4 = self.encoder(x)

        # Bottleneck
        conv5 = self.bottleneck(pool4)

        # Decoder
        conv6, conv7, conv8, conv9 = self.decoder(conv5, conv4, conv3, conv2, conv1)

        # Side Output
        convout1, convout2, convout3, convout4, finalout = self.side_output(conv6, conv7, conv8, conv9)

        return convout1, convout2, convout3, convout4, finalout
    
if __name__ == "__main__":
    
    ######## TEST ##########
    test_image = torch.torch.randn([1, 3, 400, 400])
    
    mnet = MNet()
    convout1, convout2, convout3, convout4, finalout = mnet(test_image)
    out = mnet(test_image)[:-1]
    
    print(len(out))
    
    print(f"output6 shape elem: ", convout4.shape)
    print(f"output6 shape elem: ", convout2.shape)
    print(f"output6 shape elem: ", convout2.shape)
    print(f"output6 shape elem: ", convout1.shape)
    