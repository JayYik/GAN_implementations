import torch
import torch.nn as nn
import torch.nn.functional as F
import time as t
from tqdm import tqdm
from torch import optim
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter





class GAN_D(nn.Module):

    def __init__(self, hw, channels):
        super().__init__()
        self.hw = hw
        self.dim=256

        self.main_module = nn.Sequential(

            nn.Linear(channels*hw*hw, 2*self.dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(2*self.dim, self.dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.dim, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.main_module(x)
        return x


class GAN_G(nn.Module):

    def __init__(self, hw, z_dim,channels):
        super().__init__()
        self.channels=channels
        self.hw = hw
        self.z_dim = z_dim
        self.dim = 256

        self.main_module = nn.Sequential(
            # Z latent vector
            nn.Linear(z_dim, self.dim),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Linear(self.dim, 2*self.dim),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Linear(2*self.dim, channels*hw*hw),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Tanh()
        )



    def forward(self, x):
        x = self.main_module(x)
        x = x.view(x.size(0), self.channels, self.hw, self.hw)
        return x


class GAN(nn.Module):
    def __init__(self, args):
        super(GAN, self).__init__()
        self.G=GAN_G(args.hw,args.z_dim,args.in_channels)
        self.D=GAN_D(args.hw,args.in_channels)


        self.args=args
        self.batch_size=args.batch_size
        self.z_dim=args.z_dim
        self.bce_loss = nn.BCELoss()

        self.optim_g = optim.Adam(self.G.parameters(), lr=args.lr_g,betas=args.betas)
        self.optim_d = optim.Adam(self.D.parameters(), lr=args.lr_d,betas=args.betas)

        
        # Recording program start time for log directory naming
        program_begin_time = t.strftime('%Y-%m-%d %H:%M', t.localtime())
        # Logging information
        self.information=f'GAN-{program_begin_time}'
        # TensorBoard SummaryWriter for logging
        self.writer=SummaryWriter(os.path.join(self.args.log_dir,self.information))


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
        Training function for the GAN model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            device (torch.device): The device (CPU or GPU) to perform training.

        Returns:
            None
        """

        # Move the model and loss to the specified device
        self.G.to(device)
        self.D.to(device)
        self.bce_loss.to(device)
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
                z = torch.randn((self.batch_size, self.z_dim))
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)

                # Move data to the specified device
                images=images.to(device)
                z=z.to(device)
                real_labels=real_labels.to(device)
                fake_labels=fake_labels.to(device)

                # Train Discriminator
                real_output = self.D(images)
                #print('real_output:',real_output)
                fake_images = self.G(z)
                fake_output = self.D(fake_images)
                d_real_loss = self.bce_loss(real_output.flatten(), real_labels)
                d_fake_loss = self.bce_loss(fake_output.flatten(), fake_labels)
                #print('real_loss:',d_real_loss.item(),'   fake_loss:',d_fake_loss.item())
                d_loss = d_real_loss + d_fake_loss
                self.D.zero_grad()
                d_loss.backward()
                self.writer.add_scalar('D_loss', d_loss.item(), descriminator_iter)
                self.optim_d.step()
                descriminator_iter+=1

                # Train Generator
                if i % self.args.des_iter == 0:
                    #print("i:",i)
                    self.D.zero_grad()
                    self.G.zero_grad()

                    z = torch.randn((self.batch_size, self.z_dim))
                    z = z.to(device)

                    fake_images = self.G(z)
                    fake_output = self.D(fake_images)
                    fake_score = fake_output.squeeze().mean()
                    #print('fake_output:',fake_output)
                    g_loss = self.bce_loss(fake_output.flatten(), real_labels)
                    g_loss.backward()
                    pbar.set_postfix({'G_loss': g_loss.item(),'D_loss': d_loss.item(),'fake_socre':fake_score.item()})
                    #print('g_loss:',g_loss.item())
                    self.optim_g.step()
                    self.writer.add_scalar('G_loss', g_loss.item(), generator_iter)
                    generator_iter+=1
                
                # Save generated images
                if generator_iter % 500 == 0:
                    
                    if not os.path.exists(f'./training_result_{self.args.dataset}-{self.information}/'):
                        os.makedirs(f'./training_result_{self.args.dataset}-{self.information}/')

                    z = torch.randn((self.batch_size,self.args.z_dim))
                    z=z.to(device)
                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:25]
                    grid = torchvision.utils.make_grid(samples,nrow=5)
                    torchvision.utils.save_image(grid, './training_result_{}/img_generatori_iter_{}.png'.format(self.args.dataset+'-'+self.information,str(generator_iter).zfill(3)))
            

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