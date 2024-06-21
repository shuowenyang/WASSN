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
from utils import AverageMeter
import json

from data import HStrain
from data import HStest
from SSPSR_reconstruction_inv2_SSAN2 import SSPSR
from common_WSSAN import *

# loss
from loss import HybridLoss
# from loss import HyLapLoss
from metrics import quality_assessment
from re_dataset_fc_10 import HyperDatasetValid, HyperDatasetTrain1 # Clean Data set


from tensorboardX import SummaryWriter




# global settings
resume = True
log_interval = 50
model_name = ''
test_data_dir = ''



USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:4')
else:
    device = torch.device('cpu')
print('using device:', device)

print('torch.cuda.device_count:',torch.cuda.device_count())

def main():
    # parsers
    parser = argparse.ArgumentParser(description="SSR")
    # main_parser = argparse.ArgumentParser(description="parser for SR network")
    # subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    # train_parser = subparsers.add_parser("train", help="parser for training arguments")
    # train_parser.add_argument("--cuda", type=int, required=False,default=1,
    #                           help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size, default set to 64")
    parser.add_argument("--epochs", type=int, default=40, help="epochs, default set to 20")
    parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    parser.add_argument("--n_subs", type=int, default=1, help="n_subs, default set to 8")
    parser.add_argument("--n_ovls", type=int, default=0, help="n_ovls, default set to 1")
    parser.add_argument("--n_scale", type=int, default=2, help="n_scale, default set to 2")
    parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    parser.add_argument("--dataset_name", type=str, default="CAVE", help="dataset_name, default set to dataset_name")
    parser.add_argument("--model_title", type=str, default="SSPSR", help="model_title, default set to model_title")
    parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    parser.add_argument("--save_dir", type=str, default="./trained_model_inv2_SSAN2_10_harvard/",
                              help="directory for saving trained models, default is trained_model folder")
    parser.add_argument("--gpus", type=str, default="4", help="gpu ids (default: 7)")


    opt = parser.parse_args()







    # load dataset
    print("\nloading dataset ...")
    train_data1 = HyperDatasetTrain1(mode='train')
    print("Train1:%d," % (len(train_data1),))
    val_data = HyperDatasetValid(mode='valid')
    print("Validation set samples: ", len(val_data))

    train_loader = DataLoader(dataset=train_data1, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                               pin_memory=True, drop_last=True)

    # train_loader = [train_loader1]
    eval_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)




    if opt.dataset_name=='CAVE':
        colors = 31
    elif opt.dataset_name=='Pavia':
        colors = 102
    else:
        colors = 128    

    print('===> Building model')
    # net = SSPSR(n_subs=opt.n_subs, n_ovls=opt.n_ovls, n_colors=colors, n_blocks=opt.n_blocks, n_feats=opt.n_feats, n_scale=opt.n_scale, res_scale=0.1, use_share=opt.use_share, conv=default_conv)
    net = SSPSR(n_subs=opt.n_subs, n_ovls=opt.n_ovls, n_colors=colors, n_blocks=opt.n_blocks, n_feats=opt.n_feats, res_scale=0.1, use_share=opt.use_share, conv=default_conv,device='cuda:4')

    # print(net)
    model_title = opt.dataset_name + "_" + opt.model_title +'_Blocks='+str(opt.n_blocks)+'_Subs'+str(opt.n_subs)+'_Ovls'+str(opt.n_ovls)+'_Feats='+str(opt.n_feats)
    model_name = './checkpoints_33_inv2_SSAN2_10_harvard/' +opt.dataset_name + "_" +model_title + "_ckpt_epoch_" + str(35) + ".pth"
    opt.model_title = model_title
    
    # if torch.cuda.device_count() > 1:
    #     print("===> Let's use", torch.cuda.device_count(), "GPUs.")
    #     net = torch.nn.DataParallel(net,device_ids=[6,7])

    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()

    # loss functions to choose
    mse_loss = torch.nn.MSELoss()
    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
    # hylap_loss = HyLapLoss(spatial_tv=False, spectral_tv=True)
    L1_loss = torch.nn.L1Loss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    # epoch_meter = meter.AverageValueMeter()
    writer = SummaryWriter('runs/'+model_title+'_'+str(time.ctime()))
    losses = AverageMeter()
    print('===> Start training')
    for e in range(start_epoch, opt.epochs):
        adjust_learning_rate(opt.learning_rate, optimizer, e+1)
        # epoch_meter.reset()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        for iteration, (x, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device)

            y = net(y)
            loss = h_loss(y, x)
            optimizer.zero_grad()
            # epoch_meter.add(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.parameters(), clip_para)
            optimizer.step()
            losses.update(loss.data)
            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), opt.n_blocks, opt.n_subs, opt.n_feats, opt.gpus, e+1, iteration + 1,
                                                                   len(train_loader), loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss, n_iter)

        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f}".format(time.ctime(), e+1, losses.avg))
        # run validation set every epoch
        eval_loss = validate(eval_loader, net, L1_loss)
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', losses.avg, e + 1)
        writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        # save model weights at checkpoints every 10 epochs
        if (e + 1) % 5 == 0:
            save_checkpoint(opt, net, e+1)

    # save model after training
    net.eval().cpu()
    save_model_filename = model_title + "_epoch_" + str(opt.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_') + ".pth"
    save_model_path = os.path.join(opt.save_dir, save_model_filename)
    # if torch.cuda.device_count() > 1:
    #     torch.save(net.module.state_dict(), save_model_path)
    # else:
    torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def validate(loader, model, criterion):
    # device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    # epoch_meter = meter.AverageValueMeter()
    # epoch_meter.reset()
    losses = AverageMeter()
    with torch.no_grad():
        for i, (gt, ms) in enumerate(loader):

            ms, gt = ms.to(device),  gt.to(device)
            y = model(ms)
            # x_init,y = model(ms)
            loss = criterion(y, gt)
            losses.update(loss.data)
            # epoch_meter.add(loss.item())
        mesg = "===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), losses.avg)
        print(mesg)
    # back to training mode
    model.train()
    return losses.avg

def save_checkpoint(opt, model, epoch):
    # device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints_33_inv2_SSAN2_10_harvard/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = opt.dataset_name + "_" + opt.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


if __name__ == "__main__":
    main()
