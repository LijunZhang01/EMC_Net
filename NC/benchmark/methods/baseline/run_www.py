import sys
sys.path.append('../../')
sys.path.append("/public/home/zlj/HGB_autobenchmark/NC/benchmark/scripts")

import time
import argparse
import random
from numpy import *
import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
# from GNN import myGAT
# from GNN_Res import myGAT1
from model.www import myGAT
import dgl
from sklearn.model_selection import train_test_split

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model_DBLP(args):
    feats_type = args.feats_type
    # features_list:是一个list，DBLP一共有4个值，每个值是一个array数组，第一个值shape为（4057,334）
    # 第二个值shape是（14328,4231），第三个值shape是（7723,50），第四个值shape是（20,20）
    # DBLP的adjM是一个26128*26128的稀疏矩阵
    # DBLP的labels是一个array,shape是（4057）
    # DBLP的train_val_test_idx是一个dict,分别是train，val，test的索引
    features_list, adjM, labels, train_val_test_idx, dl,emb = load_data(args.dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(type(features_list[1]))
    
    features_list = [mat2tensor(features).to(device) for features in features_list]



    emb=torch.Tensor(emb).to(device)
    if args.dataset=='ACM':
        src_node_type = 0
    else:
        src_node_type = 1
    feat_keep_idx, feat_drop_idx = train_test_split(np.arange(features_list[src_node_type].shape[0]), test_size=args.feats_drop_rate)
    type_mask=np.zeros(adjM.shape[0])
    type_mask[features_list[0].shape[0]:features_list[0].shape[0]+features_list[1].shape[0]]=1
    type_mask[features_list[0].shape[0]+features_list[1].shape[0]:features_list[0].shape[0]+features_list[1].shape[0]+features_list[2].shape[0]]=2
    type_mask[features_list[0].shape[0]+features_list[1].shape[0]+features_list[2].shape[0]:]=3
    type_mask=torch.LongTensor(type_mask)
    mask_list = []
    for i in range(len(features_list)):
        mask_list.append(np.where(type_mask == i)[0])
    for i in range(len(features_list)):
        mask_list[i] = torch.LongTensor(mask_list[i]).to(device)


    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        if(args.dataset=='ACM'):
            save = feats_type - 2
        else:
            save = 1
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    edge2type = {}
    # dl.links['data']是一个字典，代表边的集合，DBLP有6个边，所以有6个key,每一个key
    # 对应的value是一个26128*26128的稀疏矩阵
    # print(dl.links['data'])
    # for u,v in dl.links['data']:
    #     print(u)
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    # DBLP中dl.nodes['total']=26128，len(dl.links['count'])=6
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])
    

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    result_micro_f1=[]
    result_macro_f1=[]
    
    for _ in range(args.repeat):
        
        # if args.dataset=='ACM':
        #     seed(11)
        # else:
        #     seed(123)
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, 
            args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, 
            args.dropout, args.slope, True, 0.05,emb.size()[1],args.hidden_dim,args.num_heads)
        # net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, 
        # args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, 
        # args.dropout, args.slope, True, 0.05)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='./NC/benchmark/methods/baseline/checkpoint/checkpoint_{}_{}_www.pt'.format(args.dataset, args.num_layers))
        for epoch in range(args.epoch):
            
            t_start = time.time()
            # training
            net.train()
            

            logits,loss_ac = net(features_list,e_feat,feat_drop_idx,src_node_type,mask_list,feat_keep_idx,device,emb,type_mask,args,adj)
            logp = F.log_softmax(logits, 1)
            loss_classification = F.nll_loss(logp[train_idx], labels[train_idx])
            train_loss = loss_classification + args.loss_lambda*loss_ac



            # logits = net(features_list, e_feat)
            # logp = F.log_softmax(logits, 1)
            # train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | loss_classification: {:.4f}| Loss_ac: {:.4f}| Time: {:.4f}'.format(epoch, train_loss.item(), loss_classification.item(),args.loss_lambda*loss_ac.item(),t_end-t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                # logits = net(features_list, e_feat)
                # logp = F.log_softmax(logits, 1)
                # val_loss = F.nll_loss(logp[val_idx], labels[val_idx])

                logits,loss_ac = net(features_list,e_feat,feat_drop_idx,src_node_type,mask_list,feat_keep_idx,device,emb,type_mask,args,adj)
                logp = F.log_softmax(logits, 1)
                loss_classification = F.nll_loss(logp[val_idx], labels[val_idx])
                val_loss = loss_classification + args.loss_lambda*loss_ac

            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | loss_classification: {:.4f}| Loss_ac: {:.4f}| Time(s) {:.4f}'.format(
                epoch, val_loss.item(), loss_classification.item(),args.loss_lambda*loss_ac.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('./NC/benchmark/methods/baseline/checkpoint/checkpoint_{}_{}_www.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits,_ = net(features_list,e_feat,feat_drop_idx,src_node_type,mask_list,feat_keep_idx,device,emb,type_mask,args,adj)
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"./NC/benchmark/methods/baseline/{args.dataset}_{args.run}_www.txt")
            pred = onehot[pred]
            pred1=dl.evaluate(pred)
            print(pred1)
            result_micro_f1.append(pred1['micro-f1'])
            result_macro_f1.append(pred1['macro-f1'])
    print('micro:{}'.format(result_micro_f1))
    print('macro:{}'.format(result_macro_f1))
    print('ave_micro:{} std_micro:{}'.format(mean(result_micro_f1),std(result_micro_f1)))
    print('ave_macro:{} std_macro:{}'.format(mean(result_macro_f1),std(result_macro_f1)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.') # 8
    ap.add_argument('--epoch', type=int, default=500, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str,default='ACM')
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=7)
    ap.add_argument('--feats-drop-rate', type=float, default=0.4, help='The ratio of attributes to be dropped.')
    ap.add_argument('--loss-lambda', type=float, default=0.4, help='Coefficient lambda to balance loss.')

    args = ap.parse_args()
    
    # seed(11)  # ACM
    if args.dataset=='ACM':
        # seed(11)
        seed(11)
    else:
        seed(123)
    run_model_DBLP(args)
