import torch
from models.DCGAN import DCGAN
from models.GAN import GAN
from models.WGAN import WGAN_CP
from models.WGAN_GP import WGAN_GP

def get_model(args):
    if args.model == 'DCGAN':
        net=DCGAN(args)
    
    elif args.model == 'GAN':
        net=GAN(args)

    elif args.model == 'WGAN-CP':
        net=WGAN_CP(args)

    elif args.model == 'WGAN-GP':
        net=WGAN_GP(args)
    else:
        raise NotImplementedError
    return net