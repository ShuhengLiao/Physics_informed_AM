import torch
import torch.nn as nn
import time

def loss(f,target=None):
    if target == None:
        return torch.sum(torch.square(f))/f.shape[0]
    if isinstance(target,float):
        return torch.sum(torch.square(f-target))/f.shape[0]
    else:
        return nn.MSELoss()(f,target)
    
def train(net,PDE,BC,point_sets,flags,iterations=50000,lr=5e-4,info_num=100):
    params = net.parameters()
    optimizer = torch.optim.Adam(params,lr=lr)
    
    n_bc = 0
    n_ic = 0
    n_PDE = 0
    n_data =0
    for points,flag in zip(point_sets,flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_PDE += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]
            
    start_time = time.time()
    
    l_history = []
    for epoch in range(iterations):
        optimizer.zero_grad()
        l_BC = 0
        l_IC = 0
        l_PDE = 0
        l_data = 0
    
        for points,flag in zip(point_sets,flags):
            if flag[0] == 'BC':
                f = BC(points[:,0:1],points[:,1:2],points[:,2:3],points[:,3:4],flag[1])
                l_BC += loss(f)*points.shape[0]/n_bc
            if flag[0] == 'IC':
                pred = net(points)
                l_IC += loss(pred,flag[1])*points.shape[0]/n_ic
            if flag[0] == 'data':
                pred = net(points)
                l_data += loss(pred,flag[1])*points.shape[0]/n_data
            if flag[0] == 'domain':
                f = PDE(points[:,0:1],points[:,1:2],points[:,2:3],points[:,3:4],net)
                l_PDE += loss(f)*points.shape[0]/n_PDE
            
        cost = (l_BC+1e-4*l_IC+l_PDE+1e-4*l_data)/4 #weighted

        cost.backward() 
        optimizer.step()

        if n_data == 0:
            l_history.append([cost.item(),
                      l_BC.item(),
                      l_IC.item(),
                      l_PDE.item()])
        else:
            l_history.append([cost.item(),
                              l_BC.item(),
                              l_IC.item(),
                              l_PDE.item(),
                              l_data.item()])
        
        if epoch%info_num == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, L_bc: %.3e, L_ic: %.3e, L_data: %.3e, L_PDE: %.3e, Time: %.2f' 
                  % (epoch, cost, l_BC, l_IC, l_data, l_PDE,elapsed))
            start_time = time.time()
    
    return l_history

def fun():
    print('1')