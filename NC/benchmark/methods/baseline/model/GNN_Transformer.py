import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from model.conv import myGATConv
import torch.nn.functional as F
import numpy as np
from model.ac import HGNN_AC
from model.transformer import VisionTransformer
class myGAT(nn.Module):
    def __init__(self,
                
                 in_dims,
                 num_hidden,
                 attr_num,
                 embed_dim,
                 num_classes,
                 heads):
        super(myGAT, self).__init__()
        
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        
        # nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        
        self.transformer=VisionTransformer(
                              attr_num=attr_num,  #属性个数
                              attr_len=num_hidden, #原本属性长度
                              embed_dim=embed_dim,  #向量长度
                              depth=3,
                              num_heads=heads,
                              representation_size=None,
                              num_classes=num_classes)  #功能个数

    def forward(self, features_list,node_type_src):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        for i in range(len(features_list)):
            if i!=node_type_src:
                h[i]=torch.zeros_like(h[i])
                
        feature_sre=torch.cat(h, 0)
        feature_sre=feature_sre.unsqueeze(0)
        logits,embedding=self.transformer(feature_sre)
        loss_ac=torch.tensor(0)
        return logits.squeeze()[1:],loss_ac
