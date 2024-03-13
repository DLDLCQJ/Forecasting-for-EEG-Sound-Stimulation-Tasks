import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_Mome(nn.Module):
    def __init__(self,RNNs,Projection,num_layers,d_model):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        #self.attn = Attn
        #self.pj = PJ
        self.rnns = nn.ModuleList(RNNs)
        self.pj = Projection
    def forward(self,x):
        bs = x.size(0)
        device = x.device
        h0 = torch.zeros([self.num_layers,bs,self.d_model]).to(device)
        #x = self.attn(x,x,x)[0]
        # x = self.pj(x)
        h_ls=[h0]
        for i,rnn in enumerate(self.rnns):
            x,hp_pre_ = rnn(x,h_ls[i])
            h_ls.append(hp_pre_)
         
        outs = self.pj(h_ls[-1].reshape(bs,-1)).unsqueeze(-1)
        return outs
