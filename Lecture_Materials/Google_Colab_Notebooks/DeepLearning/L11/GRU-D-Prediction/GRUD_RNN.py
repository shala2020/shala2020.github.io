import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import time

class FilterMatrix(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
       
        super(FilterMatrix, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
#         print(self.weight.data)
#         print(self.bias.data)

    def forward(self, input):
#         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
        
class GRUD_RNN(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, output_last = False):
        """        
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        
        """
        
        super(GRUD_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
            self.X_mean = Variable(torch.Tensor(X_mean).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))
            self.X_mean = Variable(torch.Tensor(X_mean))
        
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        
        self.gamma_x_l = FilterMatrix(self.delta_size, self.delta_size, self.identity)
        
        self.gamma_h_l = nn.Linear(self.delta_size, self.delta_size)
        
        self.output_last = output_last
        
    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        batch_size = x.shape[0]
        dim_size = x.shape[1]

        #γ t = exp{ − max(0, W γ δ t + b γ )}
        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(delta)))
        #x̂ td = m td x td + (1 − m td ) ( γ xd t x td ′ + (1 − γ xd t ) xmean d )
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        
        #ĥ t − 1 = γ h t  h t − 1 ,
        h = delta_h * h
        
        combined = torch.cat((x, h, mask), 1)
        #z t = σ ( W z x ˆ t + U z h ˆ t − 1 + V z m t + b z )
        z = F.sigmoid(self.zl(combined))

        #r t = σ ( W r x ˆ t + U r h ˆ t − 1 + V r m t + b r )
        r = F.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        #h~ t = tanh( Wx ˆ t + U ( r t . h ˆ t − 1 ) + Vm t + b )
        h_tilde = F.tanh(self.hl(combined_r))
        #h t = (1 − z t ) . h ˆ t − 1 + z t . h~t
        h = (1 - z) * h + z * h_tilde
        
        return h
    
    def forward(self, input):
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)
        print('step_size',step_size)
        
        Hidden_State = self.initHidden(batch_size)
        X = torch.squeeze(input[:,0,:,:])
        X_last_obsv = torch.squeeze(input[:,1,:,:])
        Mask = torch.squeeze(input[:,2,:,:])
        Delta = torch.squeeze(input[:,3,:,:])
        
        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(torch.squeeze(X[:,i:i+1,:])\
                                     , torch.squeeze(X_last_obsv[:,i:i+1,:])\
                                     , torch.squeeze(self.X_mean[:,i:i+1,:])\
                                     , Hidden_State\
                                     , torch.squeeze(Mask[:,i:i+1,:])\
                                     , torch.squeeze(Delta[:,i:i+1,:]))
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
                
        if self.output_last:
            return outputs[:,-1,:]
        else:
            return outputs
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State
        





        
