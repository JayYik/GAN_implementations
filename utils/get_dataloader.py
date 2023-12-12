import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from datasets.CelebA import CelebA

def get_dataloader(args):

    if args.dataset == 'mnist':
        args.hw=32
        args.in_channels=1
        trans = transforms.Compose([
            transforms.Resize(args.hw),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=args.download, transform=trans)
        test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=args.download, transform=trans)

    elif args.dataset == 'fashion-mnist':
        args.hw=32
        args.in_channels=1
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=args.download, transform=trans)
        test_dataset = datasets.FashionMNIST(root=args.data_dir, train=False, download=args.download, transform=trans)

    elif args.dataset == 'cifar10':
        args.hw=32
        args.in_channels=3
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=args.download, transform=trans)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        args.hw=32
        args.in_channels=3
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.STL10(root=args.data_dir, split='train', download=args.download, transform=trans)
        test_dataset = datasets.STL10(root=args.data_dir,  split='test', download=args.download, transform=trans)

    elif args.dataset == 'celebA64':
        args.hw=64
        args.in_channels=3
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = CelebA(root=args.data_dir, split='train',transform=trans,resolution=64)
        test_dataset = CelebA(root=args.data_dir, split='test', transform=trans,resolution=64)

    elif args.dataset == 'celebA128':
        args.hw=128
        args.in_channels=3
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = CelebA(root=args.data_dir, split='train',transform=trans,resolution=128)
        test_dataset = CelebA(root=args.data_dir, split='test', transform=trans,resolution=128)

    elif args.dataset == 'celebA256':
        args.hw=256
        args.in_channels=3
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = CelebA(root=args.data_dir, split='train',transform=trans,resolution=256)
        test_dataset = CelebA(root=args.data_dir, split='test', transform=trans,resolution=256)
    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader