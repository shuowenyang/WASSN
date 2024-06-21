import torch
import math
import torch.nn as nn
from common_WSSAN import *

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0)):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)
        return x

class S3Dblock(nn.Module):
    def __init__(self,n_feats,convv, act, res_scale):
        super(S3Dblock, self).__init__()
        self.conv3d=nn.Sequential(
            nn.Conv3d(1, n_feats, 3, padding=3 // 2),
            BasicConv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.Conv3d(n_feats,1 , 3, padding=3 // 2),

        )
        self.cov2=nn.Conv2d(31, n_feats, 3,padding=1)
        self.SSAN=ResAttentionBlock(convv, n_feats,1, act=act, res_scale=res_scale)
        self.cov3=nn.Conv2d(256, 31, 3,padding=1)
    def forward(self,x):
        x = x.unsqueeze(1)
        x=self.conv3d(x)

        x = x.squeeze(1)
        x=self.cov2(x)
        x=self.SSAN(x)
        x=self.cov3(x)
        return x

class WSSAN(nn.Module):
    def  __init__(self,n_feats,n_blocks, convv,act, res_scale):
        super(WSSAN,self).__init__()
        block=[]
        for i in range(n_blocks):
            block.append(S3Dblock(n_feats, convv,act=act, res_scale=res_scale))
        self.net=nn.Sequential(*block)

    def forward(self,x):
        x1=self.net(x)
        # x1=self.taill(x1)

        return x1


class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, convv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(convv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(convv, n_feats, 1, act=act, res_scale=res_scale)



    def forward(self, x):

        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()

        kernel_size = 3
        m = []

        for i in range(n_blocks):
            m.append(SSB(n_feats, kernel_size, act=act, res_scale=res_scale))

        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x

        return res


# a single branch of proposed SSPSR

class GroupUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks,act, res_scale,  use_tail=True, convv=default_conv):
        super(GroupUnit, self).__init__()

        kernel_size = 3
        self.hat = convv(n_colors, n_feats, kernel_size)
        self.head = SSPN(n_feats,n_blocks,act=act, res_scale=res_scale)

        self.tail = None

        if use_tail:
            self.tail = convv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        x=self.hat(x)
        y = self.head(x)

        if self.tail is not None:
            y = self.tail(y)

        return y


class GroupUnit2(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks,act, res_scale,  use_tail=True, convv=default_conv):
        super(GroupUnit2, self).__init__()

        kernel_size = 3

        self.head = SSPN(n_feats,n_blocks,act=act, res_scale=res_scale)

        self.tail = None

        if use_tail:
            self.tail = convv(n_feats, n_colors, kernel_size)

    def forward(self, x):

        y = self.head(x)

        if self.tail is not None:
            y = self.tail(y)

        return y


class BranchUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks, act, res_scale,  use_tail=True,  convv=default_conv):
        super(BranchUnit, self).__init__()

        kernel_size = 3
        self.head = convv(n_colors, n_feats, kernel_size)
        self.body = ResAttentionBlock(convv,n_feats, n_blocks, act, res_scale)

        self.tail = None

        if use_tail:
            self.tail = convv(n_feats, n_colors, kernel_size)

    def forward(self, x):

        y = self.head(x)
        y = self.body(y)

        if self.tail is not None:
            y = self.tail(y)

        return y

class SSPSR(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_blocks, n_feats, res_scale, use_share=True, convv=default_conv,device='cuda:7'):
        super(SSPSR, self).__init__()

        self.device = device


        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        if self.shared:
            self.branch = BranchUnit(n_subs, n_feats, n_blocks, act, res_scale,  convv=default_conv)

        else:
            self.branch = nn.ModuleList()
            for i in range(self.G):
                self.branch.append(BranchUnit(n_subs, n_feats, n_blocks, act, res_scale,  convv=default_conv))

        # self.trunk2 = BranchUnit(256, n_feats, n_blocks, act, res_scale,  use_tail=False, conv=default_conv)

        #self.trunk = GroupUnit2(n_feats, n_feats, n_blocks, act, res_scale, convv=default_conv)
        self.trunk = BranchUnit(n_colors, n_feats, n_blocks, act, res_scale, use_tail=False, convv=default_conv)
        self.skip_conv = convv(n_colors, n_feats, kernel_size)
        self.final = convv(n_feats, n_colors, kernel_size)
        #self.init1 =GroupUnit(n_colors, n_feats, n_blocks, act, res_scale,  convv=default_conv)
        self.init1 = BranchUnit(n_colors, n_colors, n_blocks, act, res_scale, use_tail=False, convv=default_conv)


    def forward(self, x):

        #x = self.init1(x)

        b, c, h, w = x.shape

        # Initialize intermediate “result”, which is upsampled with n_scale//2 times
        y = torch.zeros(b, c,  33,  33)
        y = y.to(self.device)
        channel_counter = torch.zeros(c)
        channel_counter = channel_counter.to(self.device)

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            if self.shared:
                xi = self.branch(xi)
            else:
                xi = self.branch[g](xi)

            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)

        y = self.trunk(y)
        # y = y + self.skip_conv(lms)
        y = self.final(y)

        return y

