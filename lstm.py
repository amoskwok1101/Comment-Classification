import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,weight: torch.Tensor()) :
        super().__init__()    

        
        if weight is not None:
            self.embed=nn.Embedding.from_pretrained(weight,padding_idx=0,freeze=False)
        else:
            self.embed=nn.Embedding(10000,100,padding_idx=0)
        self.rnn=nn.LSTM(100,32,batch_first=True,bidirectional=True)
        self.ffn=nn.Sequential(
            nn.Linear(32*2,32),
            nn.ReLU(),
            nn.Linear(32,6),
            nn.Sigmoid())
     

        # ------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = torch.Tensor()

        # ------------------
        # Write your code here
        # The shape of x is [batch_size, length_of_sequence]
        # The shape of out is [batch_size, 1]
        out = self.embed(x)
        output,(hn,cn) = self.rnn(out)
        
        out = self.ffn(output[:,-1,:])
        

        # ------------------

        return out