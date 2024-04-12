import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
from kmeans_pytorch import kmeans


class VAE(nn.Module):
    """
    Guassian Diffusion for large-scale recommendation.
    """
    def __init__(self, item_emb, n_cate, in_dims, out_dims, device, act_func, reparam=True, dropout=0.1):
        super(VAE, self).__init__()

        self.item_emb = item_emb
        self.n_cate = n_cate
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.act_func = act_func
        self.n_item = len(item_emb)
        self.reparam = reparam
        self.dropout = nn.Dropout(dropout)
      
        in_dims_temp = [self.n_item] + self.in_dims[:-1] + [self.in_dims[-1] * 2]
        out_dims_temp = [self.in_dims[-1]] + self.out_dims + [self.n_item]

        encoder_modules = []
        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
            encoder_modules.append(nn.Linear(d_in, d_out))
            if self.act_func == 'relu':
                encoder_modules.append(nn.ReLU())
            elif self.act_func == 'sigmoid':
                encoder_modules.append(nn.Sigmoid())
            elif self.act_func == 'tanh':
                encoder_modules.append(nn.Tanh())
            else:
                raise ValueError
        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules = []
        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
            decoder_modules.append(nn.Linear(d_in, d_out))
            if self.act_func == 'relu':
                decoder_modules.append(nn.ReLU())
            elif self.act_func == 'sigmoid':
                decoder_modules.append(nn.Sigmoid())
            elif self.act_func == 'tanh':
                decoder_modules.append(nn.Tanh())
            elif self.act_func == 'leaky_relu':
                encoder_modules.append(nn.LeakyReLU())
            else:
                raise ValueError
        decoder_modules.pop()
        self.decoder = nn.Sequential(*decoder_modules)
        
        self.apply(xavier_normal_initialization)
        
    def Encode(self, batch):
        batch = self.dropout(batch)
       
        hidden = self.encoder(batch)
        mu = hidden[:, :self.in_dims[-1]]
        logvar = hidden[:, self.in_dims[-1]:]

        if self.training and self.reparam:
            latent = self.reparamterization(mu, logvar)
        else:
            latent = mu
        
        kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return batch, latent, kl_divergence
   
    def reparamterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def Decode(self, batch):
        
        return self.decoder(batch)
        
    
def compute_loss(recon_x, x):
    return -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))  


def xavier_normal_initialization(module):
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)            
                