import numpy as np
import pickle
import torch
from torch import nn
import numpy as np
import os.path
from os import path
import argparse
import math


def init_weights(m, args):
    torch.nn.init.normal_(m.weight,mean=0,std=args.sw/math.sqrt(m.in_features)) # Schoenholz et al conventions
    torch.nn.init.normal_(m.bias,mean=0,std=args.sb)

def create_networks(xs, args):
    fss = None
    for i in range(args.n_models):
        #if i % (5*10**5) == 0:
        #    print("Finished ", i, "models", datetime.datetime.now())
        if args.activation == "GaussNet":
            model = GaussNet(args)
            model.apply(lambda m: init_weights(m, args) if type(m) == nn.Linear else None)
            #print(list(model.parameters()))
        fs = model(xs)
        fs = fs.view(1,fs.shape[0],fs.shape[1])
        if fss is None:
            fss = fs 
        else:
            fss = torch.cat((fss,fs))
    return fss


class GaussNet(nn.Module):
    def __init__(self,args):
        super(GaussNet, self).__init__()
        self.args = args
        
        # create and initialize input layer
        self.input = nn.Linear(args.d_in,args.width)
        torch.nn.init.normal_(self.input.weight,mean=args.mw,std=args.sw/math.sqrt(self.input.in_features))
        torch.nn.init.normal_(self.input.bias,mean=args.mb,std=args.sb)
        
        # create and initialize output layer
        self.output = nn.Linear(args.width,args.d_out)
        torch.nn.init.normal_(self.output.weight,mean=args.mw,std=args.sw/math.sqrt(self.output.in_features))
        torch.nn.init.normal_(self.output.bias,mean=args.mb,std=args.sb)
        
    def forward(self,x):
        z = self.input(x)
        ez = torch.exp(z)
        # norm = torch.exp((4*args.sb**2+4*args.sw**2*torch.norm(x,dim=1)**2)/2.0)
        # fix args issue above
        norm = torch.exp((4+4*torch.norm(x,dim=1)**2)/(2.0*self.args.d_in))
        norm = norm.view(norm.shape[0],1)
        norm = torch.sqrt(norm)
        ezonorm = ez / norm
        return self.output(ezonorm)

def n_point(fss,n):
    num_nets, n_inputs, d_out = list(fss.shape)
    shape = [num_nets,n_inputs]
    while(len(shape)) < n+1:
        shape.append(1)
    shape.append(d_out)
    while(len(shape)) < 2*n+1:
        shape.append(1)
    fss1 = fss.view(shape)
    out = fss1
    for k in range(2,n+1):
        cur = torch.transpose(fss1,1,k)
        cur = torch.transpose(cur,1+n,k+n)
        out = out * cur
    return out
