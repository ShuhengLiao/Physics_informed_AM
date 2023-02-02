import numpy as np
import torch
import torch.nn as nn
from model import FNN
from util import *
from train import *
from torch.autograd import Variable,grad
import time
import pyvista as pv
import argparse
torch.manual_seed(0)


def output_transform(X):
    X = T_range*nn.Softplus()(X)+ T_ref
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

    t = np.linspace(x_min[3]+0.01,x_max[3],61)

    # boundary points
    bound_x_neg,_ = sampling_uniform(1.,x_min,x_max,'-x',t)
    bound_x_pos,_ = sampling_uniform(1.,x_min,x_max,'+x',t)

    bound_y_neg,_ = sampling_uniform(1.,x_min,x_max,'-y',t)
    bound_y_pos,_ = sampling_uniform(1.,x_min,x_max,'+y',t)

    bound_z_neg,_ = sampling_uniform(1.,x_min,x_max,'-z',t)
    bound_z_pos,_ = sampling_uniform(1.,x_min,x_max,'+z',t)

    bound_z_pos_more = [] # more points for surface flux
    
    for ti in t:
        if ti<=t_end:
            zi,_ = sampling_uniform(.25,
                        [max(x0+ti*v-2*r,x_min[0]),max(x_min[1],y0-2*r),x_min[2]],
                        [min(x0+ti*v+2*r,x_max[0]),min(x_max[1],y0+2*r),x_max[2]],
                        '+z',[ti])
            bound_z_pos_more.append(zi)

    bound_z_pos_more = np.vstack(bound_z_pos_more)
    bound_z_pos = np.vstack((bound_z_pos,bound_z_pos_more))

    ### domain points
    domain_pts1,_ = sampling_uniform(2.,
                                     [x_min[0],x_min[1],x_min[2]],
                                     [x_max[0],x_max[1],x_max[2]-3.],'domain',t)

    domain_pts2,_ = sampling_uniform(1.,
                                     [x_min[0],x_min[1],x_max[2]-3.+.5],
                                     [x_max[0],x_max[1],x_max[2]-1.],'domain',t)

    domain_pts3 = []
    for ti in t:
        di,_ = sampling_uniform(.5,
                                [x_min[0],x_min[1],x_max[2]-1.+.25,],
                                [x_max[0],x_max[1],x_max[2]],'domain',[ti])
        domain_pts3.append(di)
    domain_pts3 = np.vstack(domain_pts3)
    domain_pts = np.vstack((domain_pts1,domain_pts2,domain_pts3))

    # initial points
    init_pts1,_ = sampling_uniform(2.,[x_min[0],x_min[1],x_min[2]],
                                   [x_max[0],x_max[1],x_max[2]],'domain',[0],e=0)
    # more points near the toolpath origin
    init_pts2,_ = sampling_uniform(.5,[x0-2,y0-2,x_max[2]-2],
                                   [x0+2,y0+2,x_max[2]],'domain',[0])
    
    init_pts = np.vstack((init_pts1,init_pts2))
    

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


def load_data(p,f,filename,num):
    data = np.load(filename)
    if num!= 0:
        np.random.shuffle(data)
        data = data[0:num,:]
    p.extend([torch.tensor(data[:,0:4],requires_grad=True,dtype=torch.float).to(device)])
    f.extend([['data',torch.tensor(data[:,4:5],requires_grad=True,dtype=torch.float).to(device)]])
    return p,f


def BC(x,y,z,t,net,loc):
    X = torch.concat([x,y,z,t],axis=-1)
    T = net(X)
    if loc == '-x':
        T_x = grad(T,x,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return k*T_x - h*(T-T_ref) - Rboltz*emiss*(T**4-T_ref**4)
    if loc == '+x':
        T_x = grad(T,x,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return -k*T_x - h*(T-T_ref) - Rboltz*emiss*(T**4-T_ref**4)
    if loc == '-y':
        T_y = grad(T,y,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return k*T_y - h*(T-T_ref) - Rboltz*emiss*(T**4-T_ref**4)
    if loc == '+y':
        T_y = grad(T,y,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return -k*T_y - h*(T-T_ref) - Rboltz*emiss*(T**4-T_ref**4)
    if loc == '-z':
        T_t = grad(T,t,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        return T_t
    if loc == '+z':
        T_z = grad(T,z,create_graph=True,grad_outputs=torch.ones_like(T))[0]
        q = 2*P*eta/torch.pi/r**2*torch.exp(-2*(torch.square(x-x0-v*t)+torch.square(y-y0))/r**2)*(t<=t_end)*(t>0)
        return -k*T_z - h*(T-T_ref) - Rboltz*emiss*(T**4-T_ref**4) + q
    
    
if __name__ == '__main__':
    
    # augments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='GPU name')
    parser.add_argument('--output', type=str, default='bareplate', help='output filename')
    parser.add_argument('--T_ref', type=float, default=298., help='ambient temperature')
    parser.add_argument('--T_range', type=float, default=3000., help='temperature range')
    parser.add_argument('--xmax', type=float, default=40., help='max x')
    parser.add_argument('--xmin', type=float, default=0., help='min x')
    parser.add_argument('--ymax', type=float, default=10., help='max y')
    parser.add_argument('--ymin', type=float, default=0., help='min y')
    parser.add_argument('--zmax', type=float, default=6., help='max z')
    parser.add_argument('--zmin', type=float, default=0., help='min z')
    parser.add_argument('--tmax', type=float, default=3., help='max t')
    parser.add_argument('--tmin', type=float, default=0., help='min t')
    parser.add_argument('--Cp', type=float, default=.5, help='specific heat')
    parser.add_argument('--k', type=float, default=.01, help='heat conductivity')
    parser.add_argument('--x0', type=float, default=5., help='toolpath origin x')
    parser.add_argument('--y0', type=float, default=5., help='toolpath origin y')
    parser.add_argument('--r', type=float, default=1.5, help='beam radius')
    parser.add_argument('--v', type=float, default=10., help='scan speed')
    parser.add_argument('--t_end', type=float, default=3.,help='laser stop time')
    parser.add_argument('--h', type=float, default=2e-5, help='convection coefficient')
    parser.add_argument('--eta', type=float, default=.4, help='absorptivity')
    parser.add_argument('--P', type=float, default=500., help='laser power')
    parser.add_argument('--emiss', type=float, default=.3, help='emissivity')
    parser.add_argument('--rho', type=float, default=8e-3, help='rho')
    parser.add_argument('--iters', type=int, default=50000, help='number of iters')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--data', type=str, default='None', help='filename, default:None')
    parser.add_argument('--data_num', type=int, default= 0, help='number of training data used, 0 for all data')
    parser.add_argument('--calib_eta',type=bool, default = False, help='calibrate eta')
    parser.add_argument('--calib_material',type=bool, default = False, help='calibrate cp and k')
    parser.add_argument('--valid',type=str, default = '../data/1_forward/data.npy', help='validation data file')
    parser.add_argument('--pretrain',type=str, default = 'None', help='pretrained model file')
  
    args = parser.parse_args()
    
    ##############params 
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    # domain
    x_max = np.array([args.xmax, args.ymax, args.zmax, args.tmax])
    x_min = np.array([args.xmin, args.ymin, args.zmin, args.tmin])
    X_max = torch.tensor(x_max,dtype=torch.float).to(device)
    X_min = torch.tensor(x_min,dtype=torch.float).to(device)
    
    # laser params
    x0 = args.x0
    y0 = args.y0
    r = args.r
    v = args.v # speed
    t_end = args.t_end
    P = args.P # power
    eta = args.eta

    # T_ambient, and max T range
    T_ref = args.T_ref
    T_range = args.T_range

    # material params
    Cp = args.Cp
    k = args.k
    h = args.h
    Rboltz = 5.6704e-14
    emiss = args.emiss
    rho = args.rho
    
    # valid data
    data = np.load(args.valid)
    test_in = torch.tensor(data[:,0:4],requires_grad=False,dtype=torch.float).to(device)
    test_out = torch.tensor(data[:,4:5],requires_grad=False,dtype=torch.float).to(device)
    
    
    iterations = args.iters
    lr = args.lr

    net = FNN([4,64,64,64,1],nn.Tanh(),in_tf=input_transform,out_tf=output_transform)
    net.to(device)
    if args.pretrain != 'None':
        net.load_state_dict(torch.load(args.pretrain))

    point_sets,flags = generate_points([],[])
    if args.data != 'None':
        point_sets,flags = load_data(point_sets,flags,args.data,args.data_num)

    inv_params = []    
    if args.calib_eta:
        eta = torch.tensor(1e-5,requires_grad=True,device=device)
        inv_params.append(eta)
    
    if args.calib_material:
        Cp = torch.tensor(1e-5,requires_grad=True,device=device)
        inv_params.append(Cp)
        k = torch.tensor(1e-5,requires_grad=True,device=device)
        inv_params.append(k)
            
    l_history,err_history = train(net,PDE,BC,point_sets,flags,iterations,lr=lr,info_num=100,
                                        test_in = test_in,test_out = test_out,w=[1.,1e-4,1.,1e-4],
                                 inv_params = inv_params)
    
    torch.save(net.state_dict(),'../results/bareplate/{}.pt'.format( args.output))
    np.save('../results/bareplate/{}.npy'.format(args.output),l_history)
    np.save('../results/bareplate/{}_err.npy'.format(args.output),err_history)
