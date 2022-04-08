import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
from sklearn import utils
from numpy.linalg import norm

#norm from Wav2Vec
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        output = F.group_norm(
            inputs.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(inputs)
        
#wave encoder
class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fgate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.tgate = nn.Sequential(nn.Linear(channels, 1), nn.Sigmoid())

    def forward(self, x):
        
        fg = self.fgate(x.mean(dim=-1))
        x = x * fg.unsqueeze(-1)
        
        tg = x.permute(0, 2, 1).contiguous().view(-1, x.shape[1])
        tg = self.tgate(tg).view(x.shape[0], x.shape[2]).unsqueeze(1)
        out = x * tg

        return x
        
class MultiScale(nn.Module):
    def __init__(self, non_affine_group_norm=False, activation=nn.ReLU()):
        super().__init__()
        
        def block(n_in, n_out, k, stride, padding=0):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride, bias=False, padding=padding),
                Fp32GroupNorm(n_out, n_out),
                activation)
        
        self.cons = nn.ModuleList()
        
        self.cons.append(nn.Sequential(block(1, 30, 36, 18, 0), block(30, 60, 5, 1, 2)))
        self.cons.append(nn.Sequential(block(1, 30, 18, 9, 0), block(30, 50, 5, 2, 0)))
        self.cons.append(nn.Sequential(block(1, 30, 12, 6, 0), block(30, 50, 5, 3, 0)))
        
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=8)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=4)
        
        self.conv1 = block(160, 160, 5, 2)
        self.conv2 = block(160, 160, 3, 2)
        self.am1 = SEBlock(160)
        self.am2 = SEBlock(160)
        
    def forward(self, x):
        waves=[]
        max_len = []
        for conv in self.cons:
            waves.append(conv(x))
            max_len.append(conv(x).shape[-1])
        
        max_len = np.min(np.array(max_len))
        waves = torch.cat((waves[0][:,:,:max_len], waves[1][:,:,:max_len], waves[2][:,:,:max_len]), dim=1)
        
        pool1 = self.pool1(waves)
        out = self.conv1(waves)
        out = self.am1(out)
        pool2 = self.pool2(out)
        out = self.conv2(out)
        out = self.am2(out)
        
        t_max = np.min(np.array([pool1.shape[-1], pool2.shape[-1], out.shape[-1]]))
        out = torch.cat((pool1[:,:,:t_max], pool2[:,:,:t_max], out[:,:,:t_max]), dim=1)
        return out  
       
#TDNN model
class TDNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, context, dilation=1):
        super(TDNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context = context
        self.dilation = dilation
        self.kernel = nn.Linear(in_dim*context, out_dim)
        
    def forward(self, x):
        d = x.shape[1]
        x = x.unsqueeze(1)
        
        x = F.unfold(x, (self.in_dim, self.context), 
                     stride=(self.in_dim, 1), 
                     dilation=(1, self.dilation))
        
        x = x.transpose(1, 2)
        x = self.kernel(x)
        
        x = x.transpose(1, 2)

        return x
    

class TDNN_Block(nn.Module):
    def __init__(self, input_dim, output_dim=512, contex_size=5, dilation=1, norm='bn', affine=True):
        super(TDNN_Block, self).__init__()
        if norm=='bn':
            norm_layer = nn.BatchNorm1d(output_dim, affine=affine)
        elif norm == 'ln':
            norm_layer = Fp32GroupNorm(1, output_dim, affine=affine)
            
        self.tdnn_layer = nn.Sequential(
            TDNNLayer(input_dim, output_dim, contex_size, dilation),
            norm_layer,
            nn.ReLU())
        
    def forward(self, x):
        return self.tdnn_layer(x)
       
class TDNN(nn.Module):
    def __init__(self, feature_n, embed_size, norm='bn'):
        super(TDNN, self).__init__()
        self.tdnn = nn.Sequential(
            TDNN_Block(feature_n, 50, 5, 1, norm=norm),
            TDNN_Block(50, 50, 3, 2, norm=norm),
            TDNN_Block(50, 100, 1, 1, norm=norm))
        
        self.l1 = nn.Linear(200, 100)
        self.bn = nn.BatchNorm1d(100)
        self.lrelu = nn.LeakyReLU(0.2)
        self.l2 = nn.Linear(100, embed_size)
        
    def forward(self, x):
        x = self.tdnn(x)
        
        cat = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        
        x = self.lrelu(self.bn(self.l1(cat)))
        x = self.l2(x)
        
        return x
        
#final model
class final_model(nn.Module):
    def __init__(self, embed_size=100):
        super(final_model, self).__init__()
        self.multi_scale = MultiScale()
        self.tdnn_model = TDNN(480, embed_size=embed_size, norm='ln')
    
    def forward(self, x):
        out = self.multi_scale(x)
        out = self.tdnn_model(out)
        
        return out
        
#model with embed_size = 50
model = final_model(50)

#AM-softmax
class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels).item() < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
        
#loss function and optimizer
criterion = AdMSoftmaxLoss(50, 118, s=30.0, m=0.35)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#data
x = np.load('x5000.npy)
y = np.load('y5000.npy)  
x, y = utils.shuffle(x, y)
x_tensor = torch.tensor(x).view(5001, 1, -1)
y_tensor = torch.tensor(y).long()
          
#training loop
num_epochs=100
batch_size = 50
num_batches = int(x.shape[0]/batch_size)
for epoch in range(num_epochs):
    print(f'epoch: {epoch}')
    for i in range(num_batches):
        x_ = x_tensor[i*batch_size:(i+1)*batch_size]
        y_ = y_tensor[i*batch_size:(i+1)*batch_size]
        pred = model(x_)
        loss = criterion(pred, y_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch%10==0):
        print(f'step: {epoch+1}/{num_epochs}, loss: {loss}')
            
torch.save(model, 'model_50.pth')            
