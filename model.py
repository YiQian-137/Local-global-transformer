import torch
from torch import nn
import math
import numpy as np
from numpy import linalg as LA

class EmbedddingModel(nn.Module):
    def __init__(self, unit_len=[]):
        super(EmbedddingModel, self).__init__()
        self.unit_len1 = unit_len[0]
        self.unit_len2 = unit_len[1]
        self.unit_len3 = unit_len[2]
        self.unit_len4 = unit_len[3]
        self.unit_len5 = unit_len[4]

        self.embed1 = nn.Sequential(
            nn.Linear(self.unit_len1, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16, bias=True)
        )
        self.nonlinear1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64, bias=True)
        )

        self.embed2 = nn.Sequential(
            nn.Linear(self.unit_len2, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16, bias=True)
        )
        self.nonlinear2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64, bias=True)
        )

        self.embed3 = nn.Sequential(
            nn.Linear(self.unit_len3, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
             
            nn.Linear(64, 16, bias=True)
        )
        self.nonlinear3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64, bias=True)
        )

        self.embed4 = nn.Sequential(
            nn.Linear(self.unit_len4, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
              
            nn.Linear(64, 16, bias=True)
        )
        self.nonlinear4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64, bias=True)
        )

        self.embed5 = nn.Sequential(
            nn.Linear(self.unit_len5, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
              
            nn.Linear(64, 16, bias=True)
        )
        self.nonlinear5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64, bias=True)
        )
  
    def forward(self, x1, x2, x3, x4, x5):
        h1 = self.embed1(x1)
        y1 = self.nonlinear1(h1)
        h2 = self.embed2(x2)
        y2 = self.nonlinear2(h2)
        h3 = self.embed3(x3)
        y3 = self.nonlinear3(h3)
        h4 = self.embed4(x4)
        y4 = self.nonlinear4(h4)
        h5 = self.embed5(x5)
        y5 = self.nonlinear5(h5)
        hidden_state = torch.cat((h1, h2, h3, h4, h5), dim=1)  
        src = torch.stack((y1, y2, y3, y4, y5), dim=0)  
        return hidden_state, src

#LGT with noPE
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.embed = EmbedddingModel([8, 7, 6, 7, 5])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=1024, dropout=0.01,
                                                        activation="relu")
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.avg1 = nn.Linear(64, 4)
        self.linear1 = nn.Linear(4, 8, bias=True)
        self.linear2 = nn.Linear(4, 7, bias=True)
        self.linear3 = nn.Linear(4, 6, bias=True)
        self.linear4 = nn.Linear(4, 7, bias=True)
        self.linear5 = nn.Linear(4, 5, bias=True)


    def forward(self, x1, x2, x3, x4, x5):
        hidden_state, src = self.embed(x1, x2, x3, x4, x5)
        memory = self.encoder(src)  
        memory = self.avg1(memory.permute(1, 0, 2)) 
        memory = memory.permute(1, 0, 2)  
        hidden_fea1, hidden_fea2, hidden_fea3, hidden_fea4, hidden_fea5 = torch.chunk(memory, 8, dim=0) #(1,500,4)
        hidden_fea1 = torch.squeeze(hidden_fea1)  
        hidden_fea2 = torch.squeeze(hidden_fea2)
        hidden_fea3 = torch.squeeze(hidden_fea3)
        hidden_fea4 = torch.squeeze(hidden_fea4)
        hidden_fea5 = torch.squeeze(hidden_fea5)
        out = torch.cat((hidden_fea1, hidden_fea2, hidden_fea3, hidden_fea4, hidden_fea5),dim=1) #全局特征(500,20)
        fea1 = self.linear1(hidden_fea1) 
        fea2 = self.linear2(hidden_fea2)
        fea3 = self.linear3(hidden_fea3)
        fea4 = self.linear4(hidden_fea4)
        fea5 = self.linear5(hidden_fea5)
        decoder_out = torch.cat((fea1, fea2, fea3, fea4, fea5), dim=1) 
        return decoder_out, out, hidden_state


#SCPE
class PositionalEncoding(nn.Module):
    #Implement the PE function

    def __init__(self, max_len, d_model ):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        if d_model%2 ==0:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
        if d_model%2 ==1:
            pe = torch.zeros(max_len, d_model+1)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe[:,:d_model]
            pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.permute(x,(1,0,2)) 
        z = self.pe[:, :x.size(1)].clone().detach()
        x = x + z   
        return x.permute(1,0,2)  

#LGT with SCPE
class TransformerPe(nn.Module):
    def __init__(self):
        super(TransformerPE, self).__init__()
        self.embed = EmbedddingModel([8, 7, 6, 7, 5])
        self.PE = PositionalEncoding(5, 5)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=69, nhead=3, dim_feedforward=1024, dropout=0.01,
                                                        activation="relu")                                          #注意d_model和nhead的维度一致
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.avg1 = nn.Linear(69, 4)
        self.linear1 = nn.Linear(4, 8, bias=True)
        self.linear2 = nn.Linear(4, 7, bias=True)
        self.linear3 = nn.Linear(4, 6, bias=True)
        self.linear4 = nn.Linear(4, 7, bias=True)
        self.linear5 = nn.Linear(4, 5, bias=True)


    def forward(self, x1, x2, x3, x4, x5):
        hidden_state, src = self.embed(x1, x2, x3, x4, x5)
        src_pe = self.PE(src)
        memory = self.encoder(src_pe) #(5,500,128)
        memory = self.avg1(memory.permute(1, 0, 2))
        memory = memory.permute(1, 0, 2)
        hidden_fea1, hidden_fea2, hidden_fea3, hidden_fea4, hidden_fea5 = torch.chunk(memory, 8, dim=0)
        hidden_fea1 = torch.squeeze(hidden_fea1)
        hidden_fea2 = torch.squeeze(hidden_fea2)
        hidden_fea3 = torch.squeeze(hidden_fea3)
        hidden_fea4 = torch.squeeze(hidden_fea4)
        hidden_fea5 = torch.squeeze(hidden_fea5)
        out = torch.cat((hidden_fea1, hidden_fea2, hidden_fea3, hidden_fea4, hidden_fea5),dim=1) #全局特征
        fea1 = self.linear1(hidden_fea1)
        fea2 = self.linear2(hidden_fea2)
        fea3 = self.linear3(hidden_fea3)
        fea4 = self.linear4(hidden_fea4)
        fea5 = self.linear5(hidden_fea5)
        decoder_out = torch.cat((fea1, fea2, fea3, fea4, fea5), dim=1) #重构输出
        return decoder_out, out, hidden_state


#OPE
class OPE(nn.Module):
    #Implement the OPE function

    def __init__(self, max_len, d_model ):
        super(OPE, self).__init__()

        c = np.zeros((d_model, d_model))
        for i in range(d_model):
            for j in range(d_model):
                c[i,j] = 0.9**np.abs(i-j)
        np.set_printoptions(precision=4)
        w, v = LA.eig(c)
        pe = np.zeros((max_len, d_model-1))
        for i in range(max_len):
            pe[i,:] = v[i,1:]    
        pe = torch.unsqueeze(torch.from_numpy(pe),0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.permute(x,(1,0,2)) 
        z = self.pe[:, :x.size(1)].clone().detach()
        x = x + z      
        return x.permute(1,0,2).to(torch.float32)  #(5,500,128)

#LGT with OPE
class TransformerPE(nn.Module):
    def __init__(self):
        super(TransformerPE, self).__init__()
        self.embed = EmbedddingModel([8, 7, 6, 7, 5])
        self.PE = OPE(5, 65)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=1024, dropout=0.01,
                                                        activation="relu")                                          #注意d_model和nhead的维度一致
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.avg1 = nn.Linear(64, 4)
        self.linear1 = nn.Linear(4, 8, bias=True)
        self.linear2 = nn.Linear(4, 7, bias=True)
        self.linear3 = nn.Linear(4, 6, bias=True)
        self.linear4 = nn.Linear(4, 7, bias=True)
        self.linear5 = nn.Linear(4, 5, bias=True)


    def forward(self, x1, x2, x3, x4, x5):
        hidden_state, src = self.embed(x1, x2, x3, x4, x5)
        src_pe = self.PE(src)
        memory = self.encoder(src_pe) #(5,500,128)
        memory = self.avg1(memory.permute(1, 0, 2))
        memory = memory.permute(1, 0, 2)
        hidden_fea1, hidden_fea2, hidden_fea3, hidden_fea4, hidden_fea5 = torch.chunk(memory, 8, dim=0)
        hidden_fea1 = torch.squeeze(hidden_fea1)
        hidden_fea2 = torch.squeeze(hidden_fea2)
        hidden_fea3 = torch.squeeze(hidden_fea3)
        hidden_fea4 = torch.squeeze(hidden_fea4)
        hidden_fea5 = torch.squeeze(hidden_fea5)
        out = torch.cat((hidden_fea1, hidden_fea2, hidden_fea3, hidden_fea4, hidden_fea5),dim=1) 
        fea1 = self.linear1(hidden_fea1)
        fea2 = self.linear2(hidden_fea2)
        fea3 = self.linear3(hidden_fea3)
        fea4 = self.linear4(hidden_fea4)
        fea5 = self.linear5(hidden_fea5)
        decoder_out = torch.cat((fea1, fea2, fea3, fea4, fea5), dim=1) 
        return decoder_out, out, hidden_state

