import torch

def get_device(args):
    if not args.cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else "cpu")
    
    args.in_channels=1

    return device