import sys
sys.path.append('../../')
sys.path.append("/public/home/zlj/HGB_autobenchmark/NC/benchmark/scripts")
import time
import argparse
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from model.GNN_Transformer import myGAT
import dgl

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

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl,emb = load_data(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]

    attr_num=0
    for features in features_list:
        attr_num=attr_num+features.shape[0]
    emb=torch.Tensor(emb).to(device)
    src_node_type = 0
    feat_keep_idx, feat_drop_idx = train_test_split(np.arange(features_list[src_node_type].shape[0]), test_size=args.feats_drop_rate)
    
   
    in_dims = [features.shape[1] for features in features_list]
    
    labels = torch.FloatTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    

    for _ in range(args.repeat):
        loss = nn.BCELoss()
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT(in_dims, 
        args.hidden_dim, attr_num,args.hidden_dim,num_classes,args.num_heads)
        # net = torch.nn.DataParallel(net)
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
            # logits = net(features_list, e_feat)
            logp = F.sigmoid(logits)
            # print(logp[train_idx].size())
            # print(labels[train_idx].size())
            train_loss = loss(logp[train_idx], labels[train_idx])

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end-t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                logits,loss_ac = net(features_list,src_node_type)
                logp = F.sigmoid(logits)
                val_loss = loss(logp[val_idx], labels[val_idx])
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
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
            test_logits = logits[test_idx]
            pred = (test_logits.cpu().numpy()>0).astype(int)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{args.run}.txt", mode='multi')
            print(dl.evaluate(pred))


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
    ap.add_argument('--hidden-dim', type=int, default=16, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=2, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=2e-4)
    ap.add_argument('--slope', type=float, default=0.1)
    ap.add_argument('--dataset', type=str, default='IMDB')
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=10)
    ap.add_argument('--feats-drop-rate', type=float, default=0.7, help='The ratio of attributes to be dropped.')
    ap.add_argument('--loss-lambda', type=float, default=0, help='Coefficient lambda to balance loss.')

    args = ap.parse_args()
    seed(123)
    run_model_DBLP(args)
