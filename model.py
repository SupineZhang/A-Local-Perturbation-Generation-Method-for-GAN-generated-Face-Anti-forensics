import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # fake face :128*128*3
        self.conv1=nn.Conv2d(image_nc,64, kernel_size=4,stride=2,padding=0,bias=True)
        #63*63*64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2,padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        #30*30*128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        #14*14*256
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=0, bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        #6*6*512
        self.lrelu=nn.LeakyReLU(0.2)
        #
        self.conv5=nn.Conv2d(512,1,1)
        # self.fc=nn.Linear(512*6*6,2)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        out=self.conv1(x)
        out=self.lrelu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.lrelu(out)
        out=self.conv3(out)
        out = self.bn3(out)
        out = self.lrelu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.lrelu(out)
        logits=self.conv5(out)
        # out=torch.flatten(out,start_dim=1)
        # out=self.fc(out)
        out=self.sigmoid(logits)
        return logits, out

class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder=[
            #128*128*3
            nn.ReflectionPad2d(3),
            nn.Conv2d(gen_input_nc, 64, kernel_size=7, stride=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            #conv1:128*128*64
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            #conv2:64*64*128
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        ]
        res_block=[
            #32*32*256
            BasicBlock(256,256,1),
            #32*32*256
            BasicBlock(256, 256, 1),
            #32*32*256
            BasicBlock(256, 256, 1),
        ]
        decoder=[
            #128*128*3
            nn.Upsample( scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(256, 128, kernel_size=5, stride=1, bias=True),
            nn.LayerNorm(64),
            nn.ReLU(),
            #64*64*128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, bias=True),
            nn.LayerNorm(128),
            nn.ReLU(),
            #128*128*64
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, bias=True),
            nn.Tanh()
        ]

        self.encoder = nn.Sequential(*encoder)
        self.res_block = nn.Sequential(*res_block)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_block(x)
        x = self.decoder(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride,bias=True)
        self.in1 = nn.InstanceNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1,bias=True)
        self.in2 = nn.InstanceNorm2d(out_channel)
        self.pad=nn.ReflectionPad2d(1)

    def forward(self, x):
        identity = x
        out=self.pad(x)

        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)
        out=self.pad(out)
        out = self.conv2(out)
        out = self.in2(out)

        out += identity
        out = self.relu(out)

        return out


