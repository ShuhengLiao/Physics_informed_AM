import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self,layers,activation,in_tf=None,out_tf=None):
        super().__init__()
        self.activation = activation
        self.linears = nn.ModuleList()
        self.in_tf = in_tf
        self.out_tf = out_tf
        # weight initialization
        for i in range(1,len(layers)):
            self.linears.append(nn.Linear(layers[i-1],layers[i]))
            nn.init.xavier_uniform_(self.linears[-1].weight)
            nn.init.zeros_(self.linears[-1].bias)
                      
    def forward(self,inputs):
        X = inputs
        # input transformation
        if self.in_tf:
            X = self.in_tf(X)
        # linear layers    
        for linear in self.linears[:-1]:
            X = self.activation(linear(X))
        # last layer, no activation
        X = self.linears[-1](X)
        # output transformation
        if self.out_tf:
            X = self.out_tf(X)
        return X