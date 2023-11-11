import os
import numpy as np
import pickle
import pandas as pd
import scipy.sparse as sp
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from Drug_target_Luo.node2vec import node2Vec_main
from dataset import generate_data
from model import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
import train_search

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.006, help='learning rate')
parser.add_argument('--wd', type=float, default=0.09, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--dropou'
                    ''
                    't', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)
#Luo_AMG
archs = {
    "source":([[6, 1, 0]], [[9, 0, 0]]),
    "target": ([[4, 5, 1]], [[7, 1, 13]])
}

#Zheng_AMG
# archs = {
#     "source": ([[12, 12, 0]], [[12, 0, 13]]),
#     "target": ([[7, 7, 1]], [[12, 1, 13]])
# }

def main():
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    archs["source"], archs["target"] = train_search.search_main()
    steps_s = [len(meta) for meta in archs["source"][0]]
    steps_t = [len(meta) for meta in archs["target"][0]]

    datadir = "preprocessed"
    prefix = os.path.join(datadir)

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adjs_pt = []
    # Luo
    for i in range(0, 4):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    for i in range(4, 8):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_sym(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())

    # Zheng
    # for i in range(0, 5):
    #     adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
    #         normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    #     adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
    #         normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    # for i in range(5, 7):
    #     adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
    #         normalize_sym(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())

    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    # embedding
    in_dims = []
    num_drug = 0
    num_target = 0
    for k in range(num_node_types):
        in_dims.append((node_types == k).sum().item())
        if(k == 0):
            num_drug = in_dims[-1]
        elif (k == 1):
            num_target = in_dims[-1]

    node_feat = []
    data = np.load('./preprocessed/combined_matrices.npz')
    for k, d in enumerate(data):
        matrix = data[d]
        features = node2Vec_main(matrix)
        node_feat.append(torch.FloatTensor(features[:matrix.shape[0]]).cuda())
    avg_auc = []
    avg_aupr = []
    #* fold
    data_path = './data/Luo/drug_target.dat'
    pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold = generate_data(2, num_drug, data_path)
    for pos_train, pos_val, pos_test, neg_train, neg_val, neg_test in zip(pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold):
        node_feats = []
        dp_matrix = np.zeros((num_drug, num_target), dtype=int)
        dp_matrix[pos_train[:, 0], pos_train[:, 1] - num_drug] = 1
        dp_matrix[pos_val[:, 0], pos_val[:, 1] - num_drug] = 1
        features = node2Vec_main(dp_matrix)
        node_feats.append(torch.FloatTensor(features[:dp_matrix.shape[0]]).cuda())
        node_feats.append(torch.FloatTensor(features[dp_matrix.shape[0]:]).cuda())
        for fea in node_feat:
            node_feats.append(fea)

        model_s = Model(in_dims, args.n_hid, steps_s, dropout = args.dropout).cuda()
        model_t = Model(in_dims, args.n_hid, steps_t, dropout = args.dropout).cuda()
        optimizer = torch.optim.Adam(
            list(model_s.parameters()) + list(model_t.parameters()),
            lr=args.lr,
            weight_decay=args.wd
        )

        auc_best = None
        aupr_best = None
        for epoch in range(args.epochs):
            train_loss = train(node_feats, node_types, adjs_pt, pos_train, neg_train, model_s, model_t, optimizer)
            val_loss, auc_test, aupr = infer(node_feats, node_types, adjs_pt, pos_val, neg_val, pos_test, neg_test, model_s, model_t)
            if auc_best is None or auc_test > auc_best:
                auc_best = auc_test
                aupr_best = aupr
        avg_auc.append(auc_best)
        avg_aupr.append(aupr_best)
        print("AUC：{:.3f}".format(auc_best))
        print("AUPR：{:.3f}".format(aupr_best))
    print("AVG_AUC：{:.3f}".format(np.mean(avg_auc)))
    print("AVG_AUPR：{:.3f}".format(np.mean(avg_aupr)))

def train(node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, optimizer):
    model_s.train()
    model_t.train()
    optimizer.zero_grad()

    out_s = model_s(node_feats, node_types, adjs, archs["source"][0], archs["source"][1])
    out_t = model_t(node_feats, node_types, adjs, archs["target"][0], archs["target"][1])

    loss = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t):
    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, archs["source"][0], archs["source"][1])
        out_t = model_t(node_feats, node_types, adjs, archs["target"][0], archs["target"][1])

    pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
    neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
    loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))

    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.int64)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate(
        (torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))

    auc_test = roc_auc_score(y_true_test, y_pred_test)
    aupr = average_precision_score(y_true_test, y_pred_test)

    return loss.item(), auc_test, aupr

if __name__ == '__main__':
    main()