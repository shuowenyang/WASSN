import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as udata
import glob
import os
# import cv2
import random
import h5py
import numpy as np
import os
import os.path
import h5py
from scipy.io import loadmat
# import cv2
import glob
import numpy as np
import scipy.io as sio
import hdf5storage
import random
import argparse
# set up device
USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial

######################## generate train/test dataset

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')
print('using device:', device)

parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--data_path", type=str, default='../Harvard', help="data path")
parser.add_argument("--block_size", type=int, default=33, help="data patch size")
parser.add_argument("--stride", type=int, default=32, help="data patch stride,16")
parser.add_argument("--train_data_path1", type=str, default='./Dataset_33_inv_10_harvard/Valid', help="preprocess_data_path")

opt = parser.parse_args()


def main():
    if not os.path.exists(opt.train_data_path1):
        os.makedirs(opt.train_data_path1)

    Phi_33 = sio.loadmat('phi_0_10_1089.mat')

    Phi_33 = Phi_33['phi']
    # Phi_16=generate_phi_schemidt(Phi_16)
    Phi_33 = Phi_33.astype(np.float32)
    phi_16 = torch.from_numpy(Phi_33)
    process_data(block_size=opt.block_size, stride=opt.stride, mode='train',phi=phi_16)




def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

def _PRWGrayImgTensor(img,block_size,stride):

    endc = img.shape[0]
    row = img.shape[1]
    col = img.shape[2]

    if np.mod(row, block_size) == 0:
            row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)

    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)

    row_new = row - row_pad
    col_new = col - col_pad
    row_block = int(row_new / block_size)
    col_block = int(col_new / block_size)
    max_rol=row_new-block_size
    max_col = col_new - block_size
    blocknum = int((max_rol/stride+1) * (max_col/stride+1) )

    imgorg = img[:,0:row_new, 0:col_new]
    # Ipadc = imgorg / 255.0
    Ipadc = imgorg
    img_x = np.zeros([blocknum ,31, block_size, block_size], dtype=np.float32)
    count = 0
    for xi in range(0,max_rol+1,stride):
        for yj in range(0,max_col+1,stride):
            img_x[count] = Ipadc[:,xi :xi +block_size, yj:yj +  block_size]
            count = count + 1
    # img_x = torch.from_numpy(img_x)
    #
    # X = torch.empty(blocknum, 31, block_size, block_size, dtype=torch.float)
    # for i in range(X.shape[0]):
    #     X[i] = self.transform(img_x[i])

    return img_x


def Im2Patch(img,block_size,stride,phi):
    endc = img.shape[0]
    row = img.shape[1]
    col = img.shape[2]

    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)

    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)

    row_new = row - row_pad
    col_new = col - col_pad
    row_block = int(row_new / block_size)
    col_block = int(col_new / block_size)
    max_rol=row_new-block_size
    max_col = col_new - block_size
    blocknum = int((max_rol/stride+1) * (max_col/stride+1) )


    imgorg = img[:,0:row_new, 0:col_new]
    # Ipadc = imgorg / 255.0
    Ipadc = imgorg
    Ipadc = torch.from_numpy(Ipadc)
    # patches_y = np.zeros([blocknum,40, 31], dtype=np.float32)
    patches_y = np.zeros([blocknum, 33, 33,31], dtype=np.float32)
    count = 0
    for xi in range(0,max_rol+1,stride):
        for yj in range(0,max_col+1,stride):
            img_x = Ipadc[:,xi :xi +block_size, yj:yj +  block_size]
            img_x= img_x.reshape(31, 1089)
            y= torch.mm(phi, img_x.T)  # Performs a matrix-vector product
            img_inv=torch.mm(phi.T,y)
            img_inv=img_inv.view(33,33,31)
            patches_y[count]= img_inv
            # y=y.reshape(1, 1240)
            # y=torch.unsqueeze(y,0)
            # patches_y[count]=y
            count = count + 1
    patches_y=patches_y
        # X = torch.empty(blocknum, endc, block_size, block_size, dtype=torch.float)
        # for i in range(X.shape[0]):
        #     X[i] = self.transform(patches_y[i])

    return patches_y



# def Im2Patch(img, win, stride=1):
#     k = 0
#     endc = img.shape[0]
#     endw = img.shape[1]
#     endh = img.shape[2]
#     patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
#     TotalPatNum = patch.shape[1] * patch.shape[2]
#     Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
#     for i in range(win):
#         for j in range(win):
#             patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
#
#             Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
#             k = k + 1
#     return Y.reshape([endc, win, win, TotalPatNum])


def process_data(block_size, stride, mode,phi):
    if mode == 'train':
        print("\nprocess training set ...\n")
        patch_num = 1
        filenames_hyper = glob.glob(os.path.join(opt.data_path, 'Hyper_Valid_Clean', '*.mat'))

        filenames_hyper.sort()

        # for k in range(1):  # make small dataset
        for k in range(len(filenames_hyper)):
            print([filenames_hyper[k]])
            # load hyperspectral image
            # mat = h5py.File(filenames_hyper[k], 'r')
            mat = loadmat(filenames_hyper[k])
            hyper = np.float32(np.array(mat['ref']))
            hyper = np.transpose(hyper, [2, 0, 1])
            hyper = normalize(hyper, max_val=1., min_val=0.)

            # creat patches
            patches_y = Im2Patch(hyper, block_size, stride,phi)
            # patches_y = torch.from_numpy(patches_y)
            patches_hyper = _PRWGrayImgTensor(hyper, block_size,stride)
            # patches_hyper = torch.from_numpy(patches_hyper)


            for j in range(patches_hyper.shape[0]):
                print("generate training sample #%d" % patch_num)
                sub_hyper = patches_hyper[j, :, :, :]
                sub_y = patches_y[j, :, :,:]


                sub_y=np.transpose(sub_y,[2,0,1])

                train_data_path_array = [opt.train_data_path1]
                random.shuffle(train_data_path_array)
                train_data_path = os.path.join(train_data_path_array[0], 'train'+str(patch_num)+'.mat')
                hdf5storage.savemat(train_data_path, {'rad': sub_hyper}, format='7.3')
                hdf5storage.savemat(train_data_path, {'mea': sub_y}, format='7.3')
                # sio.savemat(train_data_path, {'rad': sub_hyper})
                # sio.savemat(train_data_path, {'mea': sub_y})
                patch_num += 1

        print("\ntraining set: # samples %d\n" % (patch_num-1))





if __name__ == '__main__':
    main()


