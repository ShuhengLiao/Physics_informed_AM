import numpy as np
import torch
import torch.nn as nn
from model import FNN
from util import *
from train import *
from torch.autograd import Variable,grad
import time
import argparse
torch.manual_seed(0)

def output_transform(X):
    X = T_range*nn.Softplus()(X) + T_ref
    return X

def input_transform(X):
    X = 2.*(X-X_min)/(X_max-X_min) - 1.
    return X

def PDE(x,y,t,net):
    
    X = torch.concat([x,y,t],axis=-1)
    T = net(X)
    
    T_t = grad(T,t,create_graph=True,grad_outputs=torch.ones_like(T))[0]

    T_x = grad(T,x,create_graph=True,grad_outputs=torch.ones_like(T))[0]
    T_xx = grad(T_x,x,create_graph=True,grad_outputs=torch.ones_like(T_x))[0]
    
    T_y = grad(T,y,create_graph=True,grad_outputs=torch.ones_like(T))[0]
    T_yy = grad(T_y,y,create_graph=True,grad_outputs=torch.ones_like(T_y))[0]
    
    
    Cp = a1/1000*(T-298) + b1
    k = a2*T + b2
    
    h = ha*y/1e5 + hb

    f = rho*Cp*T_t - k*(T_xx+T_yy) + 2*h/thickness*(T-T_ref) + 2*Rboltz*emiss/thickness*(T**4-T_ref**4)
    return f


def BC(x,y,t,net,loc):
    X = torch.concat([x,y,t],axis=-1)
    T = net(X)
    k = a2*T + b2
    h = ha*y/1e5 + hb
    if loc == '-x':
        T_x = grad(T,x,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return k*T_x - h*(T-T_ref) - Rboltz*emiss*(T**4-T_ref**4)
    if loc == '+x':
        T_x = grad(T,x,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return -k*T_x - h*(T-T_ref) - Rboltz*emiss*(T**4-T_ref**4)
    if loc == '-y':
        T_y = grad(T,y,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return k*T_y - hc*(T-T_ref)
    
    
    
    
def generate_points(p=[],f=[]):

    t = np.linspace(0,x_max[2],71)

    bound_x_neg,_ = sampling_uniform_2D(.5,x_min,[x_max[0],x_max[1]-1.5,x_max[2]],'-x',t)
    bound_x_pos,_ = sampling_uniform_2D(.5,x_min,[x_max[0],x_max[1]-1.5,x_max[2]],'+x',t)
    bound_y_neg,_ = sampling_uniform_2D(.5,x_min,[x_max[0],x_max[1]-1.5,x_max[2]],'-y',t)
 
    domain_pts,_ = sampling_uniform_2D([1.,.25],x_min, [x_max[0],x_max[1]-1.5,x_max[2]],'domain',t[1:],e=0.01)
    
    init_data = np.load('../data/wall/2D_init.npy').T

    p.extend([torch.tensor(bound_x_neg,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(bound_x_pos,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(bound_y_neg,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(domain_pts,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(init_data[:,0:3],requires_grad=True,dtype=torch.float).to(device)])
    f.extend([['BC','-x'],['BC','+x'],['BC','-y'],['domain'],
              ['IC',torch.tensor(init_data[:,3:4],requires_grad=True,dtype=torch.float).to(device)]])
    
    return p,f




def load_data(p=[],f=[]):

# 
    data = np.load('../data/wall/2D_IR_data.npy').T
    
    ind = (data[:,2]<7)*(data[:,1]>28.5)
    data1 = data[ind,:]
    
    if args.task != 'baseline':
        ind = (data[:,2]<7)*(data[:,1]<28.5)
        data2 = data[ind,:]
        data1 = np.vstack((data1,data2))
        
    p.extend([torch.tensor(data1[:,0:3],requires_grad=True,dtype=torch.float).to(device)])
    f.extend([['data',torch.tensor(data1[:,3:4],requires_grad=True,dtype=torch.float).to(device)]])
    return p,f


    
if __name__ == '__main__':
    # augments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='GPU name')
    parser.add_argument('--iters', type=int, default=100000, help='number of iters')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--task', type=str, default='baseline', help='baseline or calibration')
  
    args = parser.parse_args()
    
    
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
x_max = np.array([ 22.5, 30., 7.])
x_min = np.array([-22.5, 0.,  0.])
X_max = torch.tensor(x_max,dtype=torch.float).to(device)
X_min = torch.tensor(x_min,dtype=torch.float).to(device)

a1 = 2.0465e-4
b1 = 3.8091e-1
a2 = 1.6702e-5
b2 = 5.5228e-3
hc = 0.0519/15

Rboltz = 5.6704e-14
emiss = 0.2
rho = 8.19e-3

thickness = 2.5

T_ref = 298.
T_range = 3000.

net = FNN([3,64,64,64,1],nn.Tanh(),in_tf=input_transform,out_tf=output_transform)
net.to(device)

if args.task == 'baseline':
    ha = 0.
    hb = 2e-5
    a1 = torch.tensor([6.256e-2],requires_grad = True,device=device)
    b1 = torch.tensor([0.4],requires_grad = True,device=device)
    inv_params = []
else:
    ha = torch.tensor([0.],requires_grad = True,device=device)
    hb = torch.tensor([1e-5],requires_grad = True,device=device)
    a1 = torch.tensor([2.0465e-3],requires_grad = True,device=device)
    b1 = torch.tensor([0.4],requires_grad = True,device=device)
    inv_params = [a1]
    #net.load_state_dict(torch.load('../model/2Dbaseline.pt'))

iterations = args.iters



point_sets,flags = generate_points([],[])
point_sets,flags = load_data(point_sets,flags)

lr=args.lr
info_num=100
w=[1.,1e-4,1.,1e-4]

##validation data
data = np.load('../data/wall/2D_IR_data.npy').T
ind = (data[:,2]>0)*(data[:,2]<7)*(data[:,1]<28.5)
data = data[ind,:]
test_in = torch.tensor(data[:,0:3],requires_grad=False,dtype=torch.float).to(device)
test_out = torch.tensor(data[:,3:4],requires_grad=False,dtype=torch.float).to(device)

     
l_history,err_history = train2D(net,PDE,BC,point_sets,flags,iterations,lr=lr,info_num=100,
                              test_in = test_in,test_out = test_out,w=w,
                              inv_params = inv_params)


torch.save(net.state_dict(),'../model/2D{}.pt'.format(args.task))
np.save('../model/2D{}.npy'.format(args.task),l_history)
np.save('../model/2D{}_err.npy'.format(args.task),err_history)
    