import torch
import torch.nn as nn
import torch.nn.functional as F
import time as t
from tqdm import tqdm
from torch import optim
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter





class WGANCP_D(nn.Module):

    def __init__(self, hw, channels):
        super().__init__()
        self.hw = hw
        self.dim=256

        self.main_module = nn.Sequential(
            # Image (C, hw, hw)
            nn.Conv2d(in_channels=channels, out_channels=self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.LeakyReLU(0.2, inplace=True),

            # State (dim, hw//2, hw//2)
            nn.Conv2d(in_channels=self.dim, out_channels=2*self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*self.dim),
            nn.LeakyReLU(0.2, inplace=True),

            # State (2*dim, hw//4, hw//4)
            nn.Conv2d(in_channels=2*self.dim, out_channels=4*self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*self.dim),
            nn.LeakyReLU(0.2, inplace=True),

            # State (4*dim, hw//8, hw//8)
            nn.Conv2d(in_channels=4*self.dim, out_channels=4*self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*self.dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
            # outptut of main module --> State (4*dim, hw//16, hw//16)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=4*self.dim, out_channels=1, kernel_size=hw//16, stride=1, padding=0),
            # Output (1, 1, 1)
            )

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

        # weight_init
    # def weight_init(self):
    #     for m in self._modules:
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.constant_(m.bias, 0)

class WGANCP_G(nn.Module):

    def __init__(self, hw, z_dim,channels):
        super().__init__()
        self.hw = hw
        self.z_dim = z_dim
        self.dim = 256

        self.main_module = nn.Sequential(
            # Z latent vector
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=4*self.dim, kernel_size=hw//16, stride=1, padding=0),
            nn.BatchNorm2d(num_features=4*self.dim),
            nn.ReLU(True),

            # State (4*dim, hw//16, hw//16)
            nn.ConvTranspose2d(in_channels=4*self.dim, out_channels=4*self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=4*self.dim),
            nn.ReLU(True),

            # State (4*dim, hw//8, hw//8)
            nn.ConvTranspose2d(in_channels=4*self.dim, out_channels=2*self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=2*self.dim),
            nn.ReLU(True),

            # State (2*dim, hw//4, hw//4)
            nn.ConvTranspose2d(in_channels=2*self.dim, out_channels=self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.dim),
            nn.ReLU(True),

            # State (dim, hw//2, hw//2)
            nn.ConvTranspose2d(in_channels=self.dim, out_channels=channels, kernel_size=4, stride=2, padding=1),
        )
            
            # output of main module --> Image (C, hw, hw)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


        # weight_init
    # def weight_init(self):
    #     for m in self._modules:
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.constant_(m.bias, 0)

class WGAN_CP(nn.Module):
    def __init__(self, args):
        super(WGAN_CP, self).__init__()
        self.G=WGANCP_G(args.hw,args.z_dim,args.in_channels)
        self.D=WGANCP_D(args.hw,args.in_channels)
        # self.G.weight_init()
        # self.D.weight_init()

        self.args=args
        self.batch_size=args.batch_size
        self.z_dim=args.z_dim

        # Attention!!! WGAN use RMSprop optimizer instead of Adam
        self.optim_g = optim.RMSprop(self.G.parameters(), lr=args.lr_g)
        self.optim_d = optim.RMSprop(self.D.parameters(), lr=args.lr_d)
        self.scheduler_optim_g=optim.lr_scheduler.MultiStepLR(self.optim_g, milestones=[100,150], gamma=0.9)
        self.scheduler_optim_d=optim.lr_scheduler.LambdaLR(self.optim_d, lr_lambda=self.warm_up)
        
        # Recording program start time for log directory naming
        program_begin_time = t.strftime('%Y-%m-%d %H:%M', t.localtime())
        # Logging information
        self.information=f'WGAN-{program_begin_time}'
        # TensorBoard SummaryWriter for logging
        self.writer=SummaryWriter(os.path.join(self.args.log_dir,self.information))

    def warm_up(self,epoch):
        """
        Learning rate warm-up function for the RMSprop optimizer.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Adjusted learning rate based on the warm-up strategy.
        """
        top_epoch = int(self.args.num_epochs*0.3)
        if epoch<top_epoch:
            #In the first 30% of epochs, slowly increase the LR to the preset LR
            return (epoch+1) / top_epoch
        else:
            #Drop the LR to half of the preset
            return (1 -( 0.5 / (self.args.num_epochs - top_epoch) * (epoch - top_epoch) ) )

    def save_model(self,epoch):
        save_path=f'./save/{self.information}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.G.state_dict(), f'{save_path}/generator_{epoch}epochs.pth')
        torch.save(self.D.state_dict(), f'{save_path}/discriminator_{epoch}epochs.pth')
        self.save_args(save_path)
        print(f'Models save to {save_path}/generator_{epoch}epochs.pth & {save_path}/discriminator_{epoch}epochs.pth ')

    def save_args(self,save_path):
        argsDict = self.args.__dict__
        with open(f'{save_path}/setting.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')


    def train(self,train_loader,device):
        """
        Training function for the WGAN model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            device (torch.device): The device (CPU or GPU) to perform training.

        Returns:
            None
        """

        # Move the model and loss to the specified device
        self.G.to(device)
        self.D.to(device)
        generator_iter = 0
        descriminator_iter = 0
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            self.t_begin = t.time()
            pbar=tqdm(enumerate(train_loader),total=len(train_loader),ncols=100)

            for i, (images, _) in pbar:
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                 # Generate random noise and labels
                z = torch.randn((self.batch_size, self.z_dim, 1, 1))
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)

                # Move data to the specified device
                images=images.to(device)
                z=z.to(device)
                real_labels=real_labels.to(device)
                fake_labels=fake_labels.to(device)

                # Train Discriminator
                for p in self.D.parameters():
                    p.data.clamp_(-self.args.wc, self.args.wc)


                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean(0).view(1)

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean(0).view(1)

                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.writer.add_scalar('D_loss', d_loss.item(), descriminator_iter)
                self.writer.add_scalar('Wasserstein_D', Wasserstein_D.item(), descriminator_iter)
                self.optim_d.step()
                descriminator_iter+=1

                # Train Generator
                if i % self.args.des_iter == 0:
                    #print("i:",i)
                    self.D.zero_grad()
                    self.G.zero_grad()

                    z = torch.randn((self.batch_size, self.z_dim, 1, 1))
                    z = z.to(device)

                    fake_images = self.G(z)
                    
                    #print('fake_output:',fake_output)
                    g_loss = self.D(fake_images)
                    g_loss = g_loss.mean(0).view(1).mul(-1)
                    g_loss.backward()
                    pbar.set_postfix({'G_loss': g_loss.item(),'D_loss': d_loss.item()})
                    #print('g_loss:',g_loss.item())
                    self.optim_g.step()
                    self.writer.add_scalar('G_loss', g_loss.item(), generator_iter)
                    generator_iter+=1
                
                # Save generated images
                if generator_iter % 500 == 0:
                    
                    if not os.path.exists(f'./training_result_{self.args.dataset}-{self.information}/'):
                        os.makedirs(f'./training_result_{self.args.dataset}-{self.information}/')

                    z = torch.randn((self.batch_size,self.args.z_dim, 1, 1))
                    z=z.to(device)
                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:25]
                    grid = torchvision.utils.make_grid(samples,nrow=5)
                    torchvision.utils.save_image(grid, './training_result_{}/img_generatori_iter_{}.png'.format(self.args.dataset+'-'+self.information,str(generator_iter).zfill(3)))
            
            # Learning rate scheduling
            self.scheduler_optim_d.step()
            self.scheduler_optim_g.step()

            # Print and log training information
            print(self.optim_d.state_dict()['param_groups'][0]['lr'])
            print(self.optim_g.state_dict()['param_groups'][0]['lr'])
            self.t_end = t.time()
            print(
            "[Epoch %d/%d]  [D loss: %f] [G loss: %f] [training time: %.3fseconds]"
            % (epoch, self.args.num_epochs, d_loss.item(), g_loss.item() , (self.t_end - self.t_begin))
            )

            # Save the trained parameters
            if epoch % (self.args.num_epochs // 5) == 0 and epoch !=0:
                self.save_model(epoch)