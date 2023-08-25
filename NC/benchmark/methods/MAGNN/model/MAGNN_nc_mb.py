import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.base_MAGNN import MAGNN_ctr_ntype_specific
from model.ac import HGNN_AC

# support for mini-batched forward
# only support one layer for one ctr_ntype
class MAGNN_nc_mb_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_mb_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.ctr_ntype_layer = MAGNN_ctr_ntype_specific(num_metapaths,
                                                        etypes_list,
                                                        in_dim,
                                                        num_heads,
                                                        attn_vec_dim,
                                                        rnn_type,
                                                        r_vec,
                                                        attn_drop,
                                                        use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        # ctr_ntype-specific layers
        h = self.ctr_ntype_layer(inputs)

        h_fc = self.fc(h)
        return h_fc, h


class MAGNN_nc_mb(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 emb_dim,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(MAGNN_nc_mb, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        self.fc=nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_nc_mb layers
        self.layer1 = MAGNN_nc_mb_layer(num_metapaths,
                                        num_edge_type,
                                        etypes_list,
                                        hidden_dim,
                                        out_dim,
                                        num_heads,
                                        attn_vec_dim,
                                        rnn_type,
                                        attn_drop=dropout_rate)
        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=attn_vec_dim, dropout=dropout_rate,
                               activation=F.sigmoid, num_heads=num_heads)
    def forward(self, inputs,args,emb,mask_list,node_type_src,feat_keep_idx,feat_drop_idx):
        g_list, features_list, type_mask, edge_metapath_indices_list, target_idx_list = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])

        #--------------------------------------------------------------
        if args.dataset=='IMDB' or args.dataset=='ACM':
            feature_src_re = self.hgnn_ac(emb, emb[mask_list[node_type_src]],
                                    transformed_features[mask_list[node_type_src]])
            loss_ac=torch.tensor(0)
        else:
            feature_src_re = self.hgnn_ac(emb, emb[mask_list[node_type_src]][feat_keep_idx],
                                        transformed_features[mask_list[node_type_src]][feat_keep_idx])
            loss_ac = F.mse_loss(transformed_features[mask_list[node_type_src]][feat_drop_idx],
                            feature_src_re[mask_list[node_type_src]][feat_drop_idx],reduction='mean')

        if args.dataset=='DBLP':
            for i in range(len(features_list)):
                if i==node_type_src:          
                    transformed_features[mask_list[i]]+=feature_src_re[mask_list[i]]
        else:
            for i in range(len(features_list)):
                if i!=node_type_src:          
                    transformed_features[mask_list[i]]+=feature_src_re[mask_list[i]]
        if args.dataset=='DBLP':
            transformed_features=self.fc(transformed_features)
        else:
            transformed_features=transformed_features
        #--------------------------------------------------------------
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        logits, h = self.layer1((g_list, transformed_features, type_mask, edge_metapath_indices_list, target_idx_list))

        return logits, h
