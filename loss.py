import torch
import numpy as np

class HybridLoss(torch.nn.Module):
    def __init__(self, lamd=1e-1, spatial_tv=False, spectral_tv=False):
        super(HybridLoss, self).__init__()
        self.lamd = lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=1e-3)
        self.spectral = TVLossSpectral(weight=1e-3)
        self.sam=SAM_GPU(weight=1e-3)

    def forward(self, y, gt):
        # loss_init = self.fidelity(x_init, gt)
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        # sam_loss=self.sam(y,gt)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# class SAMloss(torch.nn.Module):
#     def __init__(self,weight=1.0):
#         super(SAMloss,self).__init__()
#         self.SAMLoss_weight = weight
#
#     def forward(self,y,ref):
#         (b, ch, h, w) = y.size()
#         tmp1 = y.view(b, ch, h * w).transpose(1, 2)
#         tmp2 = ref.view(b, ch, h * w)
#         sam = torch.bmm(tmp1, tmp2)
#         idx = torch.arange(0, h * w, out=torch.LongTensor())
#         sam = sam[:, idx, idx].view(b, h, w)
#         norm1 = torch.norm(y, 2, 1)
#         norm2 = torch.norm(ref, 2, 1)
#         sam = torch.div(sam, (norm1 * norm2))
#         sam = torch.sum(sam) / (b * h * w)
#         return self.SAMLoss_weight*sam




# def sam_loss(y, ref):
#     (b, ch, h, w) = y.size()
#     tmp1 = y.view(b, ch, h * w).transpose(1, 2)
#     tmp2 = ref.view(b, ch, h * w)
#     sam = torch.bmm(tmp1, tmp2)
#     idx = torch.arange(0, h * w, out=torch.LongTensor())
#     sam = sam[:, idx, idx].view(b, h, w)
#     norm1 = torch.norm(y, 2, 1)
#     norm2 = torch.norm(ref, 2, 1)
#     sam = torch.div(sam, (norm1 * norm2))
#     sam = torch.sum(sam) / (b * h * w)
#     return sam
class SAM_GPU(torch.nn.Module):
    def __init__(self,weight=1.0):
        super(SAM_GPU,self).__init__()
        self.SAMLoss_weight = weight

    # def forward(self,y,ref):

    def forward(self,im_fake, im_true):
        C = im_true.size()[0]
        H = im_true.size()[1]
        W = im_true.size()[2]
        esp = 1e-12
        Itrue = im_true.clone()#.resize_(C, H*W)
        Ifake = im_fake.clone()#.resize_(C, H*W)
        nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
        denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        sam = torch.div(nom, denominator).acos()
        sam[sam != sam] = 0
        sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
        return self.SAMLoss_weight*sam_sum