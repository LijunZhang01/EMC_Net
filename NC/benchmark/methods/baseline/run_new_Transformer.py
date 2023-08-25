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
from model.GNN_Transformer import myGAT
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
    attr_num=0
    for features in features_list:
        attr_num=attr_num+features.shape[0]


    emb=torch.Tensor(emb).to(device)
    if args.dataset=='ACM':
        src_node_type = 0
    else:
        src_node_type = 1
    feat_keep_idx, feat_drop_idx = train_test_split(np.arange(features_list[src_node_type].shape[0]), test_size=args.feats_drop_rate)
    


   
    in_dims = [features.shape[1] for features in features_list]
    
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    
    

   

    result_micro_f1=[]
    result_macro_f1=[]
    for _ in range(args.repeat):
        
        
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT(in_dims, 
        args.hidden_dim, attr_num,args.hidden_dim,num_classes,args.num_heads)
        # net = torch.nn.DataParallel(net)
        # net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, 
        # args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, 
        # args.dropout, args.slope, True, 0.05)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='./NC/benchmark/methods/baseline/checkpoint_Transformer/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()
            
            logits,loss_ac = net(features_list,src_node_type)
            # logits,loss_ac = net(features_list,e_feat,feat_drop_idx,src_node_type,mask_list,feat_keep_idx,device,emb,type_mask,args)
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
                logits,loss_ac = net(features_list,src_node_type)
                # logits,loss_ac = net(features_list,e_feat,feat_drop_idx,src_node_type,mask_list,feat_keep_idx,device,emb,type_mask,args)
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
        net.load_state_dict(torch.load('./NC/benchmark/methods/baseline/checkpoint_Transformer/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits,loss_ac = net(features_list,src_node_type)
            # logits,_ = net(features_list,e_feat,feat_drop_idx,src_node_type,mask_list,feat_keep_idx,device,emb,type_mask,args)
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"./NC/benchmark/methods/baseline/{args.dataset}_{args.run}.txt")
            pred = onehot[pred]
            pred1=dl.evaluate(pred)
            print(pred1)
            result_micro_f1.append(pred1['micro-f1'])
            result_macro_f1.append(pred1['macro-f1'])
    # print('micro:{}'.format(result_micro_f1))
    # print('macro:{}'.format(result_macro_f1))
    # print('ave_micro:{}'.format(mean(result_micro_f1)))
    # print('ave_macro:{}'.format(mean(result_macro_f1)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=16, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=2, help='Number of the attention heads. Default is 8.')
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
    ap.add_argument('--run', type=int, default=10)
    ap.add_argument('--feats-drop-rate', type=float, default=0.7, help='The ratio of attributes to be dropped.')
    ap.add_argument('--loss-lambda', type=float, default=0.1, help='Coefficient lambda to balance loss.')

    args = ap.parse_args()
    
    # seed(11)  # ACM
    seed(123)
    run_model_DBLP(args)
