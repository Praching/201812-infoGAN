import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import Gnet, Dnet, Qnet, FrontEnd, weights_init
from myUtils import save_checkpoint, load_checkpoint

class args:
    seed = 0
    cuda = True
    bs = 100
    epochs = 100
    chkpt_path = './model/'

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
                (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
        return logli.sum(1).mean().mul(-1)

class Trainer():
    def __init__(self, args):
        self.gnet = Gnet().apply(weights_init).to(device)
        self.dnet = Dnet().apply(weights_init).to(device)
        self.qnet = Qnet().apply(weights_init).to(device)
        self.frontEnd = FrontEnd().apply(weights_init).to(device)
        
        self.chkpt_path = args.chkpt_path
        self.start_epoch = 0

        self.optimizerFnD = torch.optim.Adam([
            {'params':self.frontEnd.parameters()},
            {'params':self.dnet.parameters()}], lr=2e-4, betas=(0.5, 0.99))
        
        self.optimizerGnQ = torch.optim.Adam([
            {'params':self.gnet.parameters()},
            {'params':self.qnet.parameters()}], lr=1e-3, betas=(0.5, 0.99))

        self.criterionD = nn.BCELoss()
        self.criterionQ_dis = nn.CrossEntropyLoss()
        self.criterionQ_con = log_gaussian()

        self.batch_size = args.bs
        self.class_num = 10
        self.epochs = args.epochs

        self.dataset = datasets.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, num_workers=1)


        self.cat_distribut = torch.distributions.categorical.Categorical(
            torch.ones(self.class_num)*(1.0/self.class_num))
        self.unifrom = torch.distributions.uniform.Uniform(-1, 1)


        # juet for test
        # fixed random variables
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)

        self.c1 = torch.from_numpy(np.hstack([c, np.zeros_like(c)])).to(torch.float32)
        self.c2 = torch.from_numpy(np.hstack([np.zeros_like(c), c])).to(torch.float32)

        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        self.on_hot = torch.from_numpy(one_hot).to(torch.float32)
        self.fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)

    def noise_sample(self, bs, dis_dim, con_dim=2, noise_dim=62):
        c_dis = torch.zeros(bs, dis_dim)
        samples = self.cat_distribut.sample((bs, 1))
        c_dis[range(bs),samples.view(-1)] = 1.0

        c_con = self.unifrom.sample((bs, con_dim))
        noise = self.unifrom.sample((bs, noise_dim))
        z = torch.cat([noise.clone(), c_dis.clone(), c_con.clone()], 1).view(-1, dis_dim+con_dim+noise_dim, 1, 1)
        return z, samples.view(bs), c_con.clone()

    def train(self, epoch):
        for num_iters, batch_data in enumerate(self.dataloader):
            
            self.optimizerFnD.zero_grad()
            # Training D and frontEnd
            # for real data
            self.frontEnd.train()
            self.dnet.train()
            self.gnet.train()
            self.qnet.train()

            x_real,  _ = batch_data
            x_real = x_real.to(device)
            bs = x_real.size(0)

            labels = torch.ones(bs, 1).to(device)
            fe1 = self.frontEnd(x_real)
            probs_real = self.dnet(fe1)
            loss_real = self.criterionD(probs_real, labels)
            loss_real.backward()

            # for fake data
            z, target, c_con = self.noise_sample(bs, self.class_num, con_dim=2, noise_dim=62)
            z = z.to(device)
            target = target.to(device)
            c_con = c_con.to(device)

            x_fake = self.gnet(z)
            fe2 = self.frontEnd(x_fake.detach())
            probs_fake = self.dnet(fe2)
            labels.data.fill_(0.0)
            loss_fake = self.criterionD(probs_fake, labels)
            loss_fake.backward()
            
            D_loss = loss_real + loss_fake
            self.optimizerFnD.step()

            # Training G and Q
            self.optimizerGnQ.zero_grad()
            self.frontEnd.train()
            self.dnet.train()
            self.gnet.train()
            self.qnet.train()

            fe = self.frontEnd(x_fake)
            probs_fake = self.dnet(fe)
            labels.data.fill_(1.0)
            loss_reconstruct = self.criterionD(probs_fake, labels)
            
            q_logits, q_mu, q_var = self.qnet(fe)

            loss_dis = self.criterionQ_dis(q_logits, target)
            loss_con = self.criterionQ_con(c_con, q_mu, q_var)
            
            G_loss = loss_reconstruct + loss_dis + loss_con*0.1
            G_loss.backward()
            self.optimizerGnQ.step()

            # logging
            if num_iters % 100 == 0:
                print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                    epoch, num_iters, D_loss.item(),
                    G_loss.item()
                ))
                self.gnet.eval()
                z = torch.cat([self.fix_noise, self.on_hot, self.c1], 1).view(-1, 74, 1, 1).to(device)
                x_save = self.gnet(z)
                save_image(x_save, './tmp/c1.png', nrow=10)

                z = torch.cat([self.fix_noise, self.on_hot, self.c2], 1).view(-1, 74, 1, 1).to(device)
                x_save = self.gnet(z)
                save_image(x_save.data, './tmp/c2.png', nrow=10)

    def run(self):
        self.load()
        for epoch in range(self.start_epoch, self.epochs):
            self.train(epoch)
            self.save()
            self.start_epoch += 1


    def save(self, is_best=False):
        state_dict = {
            'epoch': self.start_epoch + 1,
            'G': self.gnet.state_dict(),
            'D': self.dnet.state_dict(),
            'Q': self.qnet.state_dict(),
            'FE': self.frontEnd.state_dict(),
            'optimFnD': self.optimizerFnD.state_dict(),
            'optimGnQ': self.optimizerGnQ.state_dict(),           
        }
        save_checkpoint(state_dict, is_best, file_path=self.chkpt_path)

    
    def load(self, is_best=False):
        checkpoint = load_checkpoint(is_best, file_path=self.chkpt_path)
        if checkpoint:
            self.start_epoch = checkpoint['epoch']
            self.gnet.load_state_dict(checkpoint['G'])
            self.dnet.load_state_dict(checkpoint['D'])
            self.qnet.load_state_dict(checkpoint['Q'])
            self.frontEnd.load_state_dict(checkpoint['FE'])
            self.optimizerFnD.load_state_dict(checkpoint['optimFnD'])
            self.optimizerGnQ.load_state_dict(checkpoint['optimGnQ'])

            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))

        

if __name__ == "__main__":
    test_train = Trainer(args)
    test_train.run()
