import argparse
import os
import sys
import random
import time
import torch
import cv2 as cv
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from torchnet import meter
# from utils import AverageMeter
import json
from re_utils_utils import reconstruction_whole_image_cpu, save_matv73
from data import HStrain
from data import HStest
# from mix_model_attention import MCNet
from SSPSR_reconstruction_inv2_SSAN2 import SSPSR
from common_WSSAN import *
import scipy.io as sio
import hdf5storage
# loss
import scipy.io
from loss import HybridLoss
# from loss import HyLapLoss
from metrics import quality_assessment
from re_dataset_fc_10 import HyperDatasetValid, HyperDatasetTrain1,HyperDatasetTest,HyperDatasetTest2 # Clean Data set


from tensorboardX import SummaryWriter
# test_data_dir = './test_image/'+'.mat'
USE_GPU = True

model_path = '/media/omnisky/184a7d04-f9dd-48aa-807f-81b547b91836/ysw/DeepInverse-Pytorch/SSPSR-master/trained_model_inv2_SSAN2_10_harvard/CAVE_SSPSR_Blocks=3_Subs1_Ovls0_Feats=256_epoch_40_Thu_Jul__1_20:39:26_2021.pth'
# model_path = '/media/omnisky/184a7d04-f9dd-48aa-807f-81b547b91836/ysw/DeepInverse-Pytorch/SSPSR-master/trained_model_33_mix/CAVE_SSPSR_Blocks=3_Subs1_Ovls0_Feats=256_epoch_20_Tue_Jan_19_07:30:24_2021.pth'

result_path = './test_results/harvard/0.10'
# img_path = '/media/omnisky/184a7d04-f9dd-48aa-807f-81b547b91836/ysw/DeepInverse-Pytorch/SSPSR-master/test_image/'
var_name = 'cube'
block_size=33


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def test(csrate):



    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # device = torch.device("cuda" if args.cuda else "cpu")
    Phi_data_Name = 'phi_0_%s_1089.mat' % csrate[2:4]
    Phi_data = sio.loadmat(Phi_data_Name)

    # Phi_64 = sio.loadmat('phi_0_%s_1089.mat'% (csrate)[2:4])
    Phi_data = Phi_data['phi']
    Phi_data = Phi_data.astype(np.float32)
    phi= torch.from_numpy(Phi_data)




    print('===> Loading testset')
    test_set = HyperDatasetTest2(mode='test') ########### load test image
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    with torch.no_grad():

        # loading model
        test_number = 0
        ####inv
        # model = SSPSR(n_subs=1, n_ovls=0, n_colors=31, n_blocks=3, n_feats=256, res_scale=0.1, use_share=True, conv=default_conv)
        ####inv2,SSAN,SSAN2,,,
        model = SSPSR(n_subs=1, n_ovls=0, n_colors=31, n_blocks=3, n_feats=256, res_scale=0.1, use_share=True, conv=default_conv,device='cuda:0')
        # model=MCNet(upscale_factor=1,n_colors=31,n_feats=128,n_conv=1,device='cuda:0')
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.to(device).eval()

        for t,  ms  in enumerate(test_loader):

            row = ms.shape[2]
            col = ms.shape[3]

            if np.mod(row, block_size) == 0:
                row_pad = 0
            else:
                row_pad = block_size - np.mod(row, block_size)

            if np.mod(col, block_size) == 0:
                col_pad = 0
            else:
                col_pad = block_size - np.mod(col, block_size)
            row_new = row + row_pad
            col_new = col + col_pad

            max_rol = row_new - block_size
            max_col = col_new - block_size

            ms=ms[:,:,0:max_rol,0:max_col]

            Rec = np.zeros(np.shape(ms))
            Rec = torch.from_numpy(Rec)
            pred_time = []
            indices={}
            i = 0
            count = 0

            while i + 33 <= max_rol:
                j = 0
                while j + 33 <=  max_col:
                    x = ms[:, :,i:i + 33, j:j + 33]
                    # x = torch.from_numpy(x)
                    Rec = Rec.to(device)
                    # y=np.transpose(x,[2,0,1])

                    input = x.reshape(31, 1089)
                    # input = input.astype(np.float32)
                    y = torch.mm(phi, input.T)  # Performs a matrix-vector product
                    # y = np.expand_dims(y, axis=0)
                    # input = np.expand_dims(y, axis=0)
                    # input=input.transpose[0,3,2,1]
                    phi_inv = phi.T
                    input = torch.mm(phi_inv, y)
                    input = input.view(33, 33, 31)
                    input = input.permute(2,0,1)
                    input=input.unsqueeze(dim=0)
                    input = input.to(device)

                    t = time.time()

                    y = model(input)
                    Rec[:,:,i:i + 33, j:j + 33 ] = y

                    print(time.time() - t)
                    pred_time.append(time.time() - t)
                    count += 1
                    j += 33
                i += 33
            print(sum(pred_time))
            Rec, ms = Rec.squeeze().cpu().numpy().transpose(1, 2, 0), ms.squeeze().cpu().numpy().transpose(1, 2, 0)
            result = Rec[:ms.shape[0], :ms.shape[1], :]

            # # compute quality
            indices = quality_assessment(ms, result, data_range=1., ratio=4)
            test_number += 1
            print(indices)

            # if t==0:
            #     indices = quality_assessment(ms, result, data_range=1., ratio=4)
            # else:
            #     indices = sum_dict(indices, quality_assessment(ms, result, data_range=1., ratio=4))
            # 
            # test_number += 1
            # for index in indices:
            #     indices[index] = indices[index] / test_number
            # 
            # print(indices)

            result_data_path = [result_path]

            mat_dir = os.path.join(result_data_path[0], 'test_inv2_SSAN2_w_%s' % csrate[2:4]+ '_'+str(test_number) + '.mat')

            hdf5storage.savemat(mat_dir, {'result': result}, format='7.3')
            hdf5storage.savemat(mat_dir, {'gt': ms}, format='7.3')


    print("Test finished")




if __name__ == '__main__':
    csrate = '0.10'
    test(csrate)