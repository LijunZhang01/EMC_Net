import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from model.conv import myGATConv

import torch.nn.functional as F
import numpy as np
from model.ac_one import HGNN_AC
class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 emb_dim,
                 attn_vec_dim,
                 num_heads,
                 dropout_rate=0.5,
                 cuda=False):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.hidden_dim = num_hidden
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.fc=nn.Linear(num_hidden, num_hidden, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.bu_gat=myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, 1,
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=attn_vec_dim, dropout=dropout_rate,
                               activation=F.sigmoid, num_heads=num_heads, cuda=cuda)

    def forward(self, features_list,e_feat,feat_drop_idx,node_type_src,mask_list,feat_keep_idx,device,emb,type_mask,args,adj):
        


        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        #----------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------
        feat_src = h
        feature_src_re = self.hgnn_ac(adj[mask_list[node_type_src]][:, mask_list[node_type_src]][:, feat_keep_idx],
                                      emb[mask_list[node_type_src]], emb[mask_list[node_type_src]][feat_keep_idx],
                                      feat_src[mask_list[node_type_src]][feat_keep_idx])
        loss_ac = F.mse_loss(feat_src[mask_list[node_type_src]][feat_drop_idx], feature_src_re[feat_drop_idx, :])
        
        for i in range(len(features_list)):
            if i!=node_type_src:          
                feat_ac = self.hgnn_ac(adj[mask_list[i]][:, mask_list[node_type_src]],
                                    emb[mask_list[i]], emb[mask_list[node_type_src]],
                                    feat_src[mask_list[node_type_src]])
                h[mask_list[i]] = feat_ac
        

        # h = self.feat_drop(h)


        #----------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------


        #----------------------------------------------------------------------------------------------
        # feat_src = h

        
        # # attribute completion
        
        # if args.dataset=='IMDB' or args.dataset=='ACM':
        #     feature_src_re = self.hgnn_ac(emb, emb[mask_list[node_type_src]],
        #                             feat_src[mask_list[node_type_src]])
        #     loss_ac=torch.tensor(0)
        # else:
        #     feature_src_re = self.hgnn_ac(emb, emb[mask_list[node_type_src]][feat_keep_idx],
        #                                 feat_src[mask_list[node_type_src]][feat_keep_idx])
        #     loss_ac = F.mse_loss(feat_src[mask_list[node_type_src]][feat_drop_idx],
        #                     feature_src_re[mask_list[node_type_src]][feat_drop_idx],reduction='mean')
          


        # if args.dataset=='DBLP':
        #     for i in range(len(features_list)):
        #         # feat_src[mask_list[i]]+=feature_src_re[mask_list[i]]
        #         if i==node_type_src:          
        #             feat_src[mask_list[i]]+=feature_src_re[mask_list[i]]
        #     # for i in range(len(features_list)):
        #     #     if i!=node_type_src:          
        #     #         feat_src[mask_list[i]]+=feature_src_re[mask_list[i]]
        # else:
        #     for i in range(len(features_list)):
        #         if i!=node_type_src:          
        #             feat_src[mask_list[i]]+=feature_src_re[mask_list[i]]
        
       
        # if args.dataset=='DBLP':
        #     h=self.fc(feat_src)
        # else:
        #     h=feat_src
        
        #----------------------------------------------------------------------------------------------
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)

        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        # loss_ac=torch.tensor(0)
        return logits,loss_ac
