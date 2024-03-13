import torch.nn as nn


from layers.RNN_Mome import RNN_Mome

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
       
        self.decoder =  RNN_Mome(
                       
                        [nn.GRU(
                        input_size = configs.c_out if l == configs.u_layers-1 else configs.d_model,
                        hidden_size = configs.d_model,
                        batch_first=True,
                        ) for l in reversed(range(configs.u_layers))],

                        nn.Sequential(nn.Linear(
                        configs.d_model,
                        configs.pred_len,
                        bias=True),
                        #nn.Sigmoid()
                        ),
                        configs.num_layers,
                        configs.d_model
                        )       
                               
    def forward(self, x):
       
        dec_out = self.decoder(x)
        return dec_out
