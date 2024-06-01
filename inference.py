import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as tfs
import numpy as np
import os
from PIL import Image


import torch.nn as nn
import torch.nn.functional as F
import torch


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class BlockUNet1(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, relu=False, drop=False, bn=True):
        super(BlockUNet1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

        self.dropout = nn.Dropout2d(0.5)
        self.batch = nn.InstanceNorm2d(out_channels)

        self.upsample = upsample
        self.relu = relu
        self.drop = drop
        self.bn = bn

    def forward(self, x):
        if self.relu:
            y = F.relu(x)
        elif not self.relu:
            y = F.leaky_relu(x, 0.2)
        if self.upsample:
            y = self.deconv(y)
            if self.bn:
                y = self.batch(y)
            if self.drop:
                y = self.dropout(y)

        elif not self.upsample:
            y = self.conv(y)
            if self.bn:
                y = self.batch(y)
            if self.drop:
                y = self.dropout(y)

        return y


class G2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G2, self).__init__()

        self.conv = nn.Conv2d(in_channels, 8, 4, 2, 1, bias=False)
        self.layer1 = BlockUNet1(8, 16)
        self.layer2 = BlockUNet1(16, 32)
        self.layer3 = BlockUNet1(32, 64)
        self.layer4 = BlockUNet1(64, 64)
        self.layer5 = BlockUNet1(64, 64)
        self.layer6 = BlockUNet1(64, 64)
        self.layer7 = BlockUNet1(64, 64)
        self.dlayer7 = BlockUNet1(64, 64, True, True, True, False)
        self.dlayer6 = BlockUNet1(128, 64, True, True, True)
        self.dlayer5 = BlockUNet1(128, 64, True, True, True)
        self.dlayer4 = BlockUNet1(128, 64, True, True)
        self.dlayer3 = BlockUNet1(128, 32, True, True)
        self.dlayer2 = BlockUNet1(64, 16, True, True)
        self.dlayer1 = BlockUNet1(32, 8, True, True)
        self.relu = nn.ReLU()
        self.dconv = nn.ConvTranspose2d(16, out_channels, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.layer1(y1)
        y3 = self.layer2(y2)
        y4 = self.layer3(y3)
        y5 = self.layer4(y4)
        y6 = self.layer5(y5)
        y7 = self.layer6(y6)

        dy7 = self.dlayer7(y7)
        # print('dy7: {} y6: {}', dy7.shape, y6.shape)
        concat6 = torch.cat([dy7, y6], 1)
        dy6 = self.dlayer5(concat6)
        concat5 = torch.cat([dy6, y5], 1)
        dy5 = self.dlayer4(concat5)
        concat4 = torch.cat([dy5, y4], 1)
        dy4 = self.dlayer3(concat4)
        concat3 = torch.cat([dy4, y3], 1)
        dy3 = self.dlayer2(concat3)
        concat2 = torch.cat([dy3, y2], 1)
        dy2 = self.dlayer1(concat2)
        concat1 = torch.cat([dy2, y1], 1)
        out = self.relu(concat1)
        out = self.dconv(out)
        out = self.lrelu(out)

        return F.avg_pool2d(out, (out.shape[2], out.shape[3]))


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class FFA(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

        self.J_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.Conv2d(32, 3, 3, 1, 1, bias=False)
        )
        self.T_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.Conv2d(32, 3, 3, 1, 1, bias=False)
        )
        self.T_var = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.Conv2d(32, 3, 3, 1, 1, bias=False)
        )

        self.ANet = G2(3, 3)

    def forward(self, x1, x2=0, Val=False, stage='stage1'):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)

        out_J = self.J_head(out) + x1
        out_T = self.T_head(out)
        if stage == 'stage1':
            T_var = self.T_var(out)
        else:
            with torch.no_grad():
                T_var = self.T_var(out)

        if not Val:
            out_A = self.ANet(x1)
        else:
            out_A = self.ANet(x2)

        out_I = out_T * out_J + (1 - out_T) * out_A

        # x = self.post(out)
        return out, out_J, out_T, T_var, out_A, out_I


def prepare_data(hazy_name):
    hazy = Image.open(hazy_name)
    if hazy.mode != 'RGB':
        hazy = hazy.convert('RGB')
    hazy1 = tfs.Compose([
        tfs.ToTensor(),
        # tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(hazy)[None, ::]
    return hazy1


def load_check(name=None):
    names = ['NHHaze', 'Dense', 'its']
    assert name in names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FFA(gps=3, blocks=19)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    ckp_name = './net/' + name + '_train_ffa_3_19.pk'
    ckp = torch.load(ckp_name)
    net.load_state_dict(ckp['model'])
    net.eval()
    return net

def test(img, model):
    img = img.cuda()
    hazy1 = tfs.Compose([
        tfs.ToTensor(),
        # tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(img)[None, ::]
    hazy_for_A = F.interpolate(hazy1, size=(256, 256), mode='bilinear')
    _, clean, _, _, _, _ = model(hazy1, hazy_for_A, True)
    return clean


if __name__ == '__main__':
    # device_ids = [Id for Id in range(torch.cuda.device_count())]
    names = ['NHHaze', 'Dense', 'its']
    model = load_check(names[0])
    # input_img = input()
    # dehazed = test(input_img, model)

