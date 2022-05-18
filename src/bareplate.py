import numpy as np
import torch
import torch.nn as nn
from model import FNN
from util import *
from train import *
from torch.autograd import Variable,grad
import time

def output_transform(X):
    X = T_range*nn.Softplus()(X) + T_ref
    return X


def input_transform(X):
    X = 2.*(X-X_min)/(X_max-X_min) - 1.
    return X


def PDE(x,y,z,t,net):
    X = torch.concat([x,y,z,t],axis=-1)
    T = net(X)
    
    T_t = grad(T,t,create_graph=True,grad_outputs=torch.ones_like(T))[0]

    T_x = grad(T,x,create_graph=True,grad_outputs=torch.ones_like(T))[0]
    T_xx = grad(T_x,x,create_graph=True,grad_outputs=torch.ones_like(T_x))[0]
    
    T_y = grad(T,y,create_graph=True,grad_outputs=torch.ones_like(T))[0]
    T_yy = grad(T_y,y,create_graph=True,grad_outputs=torch.ones_like(T_y))[0]
    
    T_z = grad(T,z,create_graph=True,grad_outputs=torch.ones_like(T))[0]
    T_zz = grad(T_z,z,create_graph=True,grad_outputs=torch.ones_like(T_z))[0]
    
    f = rho*Cp*T_t - k*(T_xx+T_yy+T_zz)

    return f


def generate_points(p=[],f=[]):

    t = np.linspace(0,x_max[3],31)

    # boundary points
    bound_x_neg,_ = sampling_uniform(1.,x_min,x_max,'-x',t)
    bound_x_pos,_ = sampling_uniform(1.,x_min,x_max,'+x',t)

    bound_y_neg,_ = sampling_uniform(1.,x_min,x_max,'-y',t)
    bound_y_pos,_ = sampling_uniform(1.,x_min,x_max,'+y',t)

    bound_z_neg,_ = sampling_uniform(1.,x_min,x_max,'-z',t)
    bound_z_pos,_ = sampling_uniform(1.,x_min,x_max,'+z',t)

    bound_z_pos_more = [] # more points for surface flux
    for ti in t:
        if ti<t_end:
            zi,_ = sampling_uniform(.25,
                        [max(x0+ti*v-2*r,x_min[0]),max(x_min[1],y0-2*r),x_min[2]],
                        [min(x0+ti*v+2*r,x_max[0]),min(x_max[1],y0+2*r),x_max[2]],
                        '+z',[ti])
            bound_z_pos_more.append(zi)

    bound_z_pos_more = np.vstack(bound_z_pos_more)
    bound_z_pos = np.vstack((bound_z_pos,bound_z_pos_more))

    ### domain points
    e = 0.05
    domain_pts1,_ = sampling_uniform(2.,
                                     [x_min[0]+e,x_min[1]+e,x_min[2]+e],
                                     [x_max[0]-e,x_max[1]-e,x_max[2]-3.],'domain',t)

    domain_pts2,_ = sampling_uniform(1.,
                                     [x_min[0]+e,x_min[1]+e,x_max[2]-3.+e],
                                     [x_max[0]-e,x_max[1]-e,x_max[2]-1.],'domain',t)

    domain_pts3 = []
    for ti in t:
        di,_ = sampling_uniform(.5,
                                [x_min[0]+e,x_min[1]+e,x_max[2]-1.+e,],
                                [x_max[0]-e,x_max[1]-e,x_max[2]-e],'domain',[ti])
        domain_pts3.append(di)
    domain_pts3 = np.vstack(domain_pts3)
    domain_pts = np.vstack((domain_pts1,domain_pts2,domain_pts3))

    init_pts,_ = sampling_uniform(2.,[x_min[0],x_min[1],x_min[2]],
                                   [x_max[0],x_max[1],x_max[2]],'domain',[0])

    p.extend([torch.tensor(bound_x_neg,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(bound_x_pos,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(bound_y_neg,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(bound_y_pos,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(bound_z_neg,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(bound_z_pos,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(init_pts,requires_grad=True,dtype=torch.float).to(device),
              torch.tensor(domain_pts,requires_grad=True,dtype=torch.float).to(device)])
    f.extend([['BC','-x'],['BC','+x'],['BC','-y'],['BC','+y'],['BC','-z'],['BC','+z'],['IC',T_ref],['domain']])
    
    return p,f


def BC(x,y,z,t,loc,l=None):
    X = torch.concat([x,y,z,t],axis=-1)
    T = net(X)
    if loc == '-x':
        T_x = grad(T,x,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return k*T_x - h*T - Rboltz*emiss*T**4
    if loc == '+x':
        T_x = grad(T,x,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return -k*T_x - h*T - Rboltz*emiss*T**4
    if loc == '-y':
        T_y = grad(T,y,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return k*T_y - h*T - Rboltz*emiss*T**4
    if loc == '+y':
        T_y = grad(T,y,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return -k*T_y - h*T - Rboltz*emiss*T**4
    if loc == '-z':
        T_t = grad(T,t,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return T_t
    if loc == '+z':
        T_z = grad(T,z,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        q = 2*P*eta/torch.pi/r**2*torch.exp(-2*(torch.square(x-x0-v*t)+torch.square(y-y0))/r**2)*(t<=t_end)
        return -k*T_z - h*T + q

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    task = 'bareplate'
    x_max = np.array([40., 10.,  6.,  3.])
    x_min = np.array([ 0.,  0.,  0.,  0.])
    X_max = torch.tensor(x_max,dtype=torch.float).to(device)
    X_min = torch.tensor(x_min,dtype=torch.float).to(device)

    T_ref = 298.
    T_range = 3000.

    Cp = 0.5
    k = 0.01

    x0 = 5. # laser start x
    y0 = 5.
    r = 1.5 # beam radius
    v = 10 # speed
    t_end = 3. # laser end time

    h = 2e-5
    eta = 0.4
    P = 500
    Rboltz = 5.6704e-14
    emiss = 0.3
    rho = 8e-3

    iterations = 50000

    net = FNN([4,64,64,64,1],nn.Tanh(),in_tf=input_transform,out_tf=output_transform)
    net.to(device)

    point_sets,flags = generate_points([],[])

    l_history = train(net,PDE,BC,point_sets,flags,iterations,lr=5e-4,info_num=100)

    torch.save(net.state_dict(),'../model/{}.pt'.format(task))
    np.save('../model/{}.npy'.format(task),l_history)