import argparse
import os
import utils
import torch
import numpy as np
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/", help="data directory")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="log directory")
    parser.add_argument("--model_dir", type=str, default="models/", help="model directory")
    parser.add_argument("--result_dir", type=str, default="results/", help="result directory")
    parser.add_argument("--cuda", type=bool, default=True, help="if use cuda or cpu")
    parser.add_argument("--gpu_ids",type=int,default=0,help="gpu ids")
    parser.add_argument("--download",type=bool,default=True,help="download dataset")
    parser.add_argument("--seed",type=int,default=114514,help="random seed")

    parser.add_argument('--model',type=str,default='GAN',choices=['DCGAN','GAN','WGAN-CP','WGAN-GP'],
                        help='model name')
    parser.add_argument('--dataset',type=str,default='mnist',choices=['celebA64','mnist','fashion-mnist',
                                                                       'cifar10','stl10','celebA128','celebA256',])

    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--betas",type=float,default=(0.5,0.999),help="betas")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--pic_channels",type=int,dest='in_channels',default=3,help="input channels")
    parser.add_argument("--img_size",type=int,dest='hw',default=64,help="image size")
    parser.add_argument("--z_dim",type=int,dest='z_dim',default=100,help="latent variable z dimension")
    parser.add_argument("--des_iter",type=int,default=1,help="number of iterations for the discriminator")
    parser.add_argument("--wc",type=float,default=0.01,help="range of weight clipping")
    parser.add_argument("--gp_lambda",type=float,default=10,help="gradient penalty lambda hyperparameter")

    args=parser.parse_args()

    return args

def set_random_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def main():


    args=get_args() #get arguments
    set_random_seed(args) #fix the random seed
    device=utils.get_device(args) #get device
    train_dataloader,test_dataloader=utils.get_dataloader(args)

    net=utils.get_model(args) #get model
    net.train(train_dataloader,device)

if __name__ == '__main__':
    main()