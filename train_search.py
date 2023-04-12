import os
import numpy as np
import pickle
import scipy.sparse as sp
import argparse
import torch
import torch.nn.functional as F

from Drug_target_Luo.dataset import generate_data
from Drug_target_Luo.node2vec import node2Vec_main
from model_search import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor

"""执行搜索的脚本"""
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for supernet training')
parser.add_argument('--eps', type=float, default=0., help='probability of random sampling')
parser.add_argument('--decay', type=float, default=0.8, help='decay factor for eps')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + \
         "_h" + str(args.n_hid) + "_alr" + str(args.alr) + "_epoch" + str(args.epochs) + \
         "_cuda" + str(args.gpu) + "_eps" + str(args.eps) + "_d" + str(args.decay)

cstr_source = [0]

cstr_target = [1]
def search_main():
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

    # embedding
    in_dims = []
    num_drug = 0
    num_target = 0
    for k in range(num_node_types):
        in_dims.append((node_types == k).sum().item())
        if (k == 0):
            num_drug = in_dims[-1]
        elif (k == 1):
            num_target = in_dims[-1]

    path = './data/Luo/drug_target.dat'
    pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold = generate_data(1, num_drug, path)
    pos_train = pos_train_fold[0]
    pos_val = pos_val_fold[0]
    pos_test = pos_test_fold[0]
    neg_train = neg_train_fold[0]
    neg_val = neg_val_fold[0]
    neg_test = neg_test_fold[0]

    data = np.load('./preprocessed/combined_matrices.npz')
    node_feats = []
    dp_matrix = np.zeros((num_drug, num_target), dtype=int)
    dp_matrix[pos_train[:, 0], pos_train[:, 1] - num_drug] = 1
    dp_matrix[pos_val[:, 0], pos_val[:, 1] - num_drug] = 1
    features = node2Vec_main(dp_matrix)
    node_feats.append(torch.FloatTensor(features[:dp_matrix.shape[0]]).cuda())
    node_feats.append(torch.FloatTensor(features[dp_matrix.shape[0]:]).cuda())

    for k, d in enumerate(data):
        matrix = data[d]
        features = node2Vec_main(matrix)
        node_feats.append(torch.FloatTensor(features[:matrix.shape[0]]).cuda())

    t = [4]
    model_s = Model(in_dims, args.n_hid, len(adjs_pt), t, cstr_source).cuda()
    model_t = Model(in_dims, args.n_hid, len(adjs_pt), t, cstr_target).cuda()

    optimizer_w = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )

    optimizer_a = torch.optim.Adam(
        model_s.alphas() + model_t.alphas(),
        lr=args.alr
    )

    eps = args.eps
    minval_error = None
    for epoch in range(args.epochs):
        train_error, val_error = train(node_feats, node_types, adjs_pt, pos_train, neg_train, pos_val, neg_val, model_s, model_t, optimizer_w, optimizer_a, eps)
        if (minval_error == None or minval_error > val_error):
            minval_error = val_error
            s = model_s.parse()
            t = model_t.parse()
        eps = eps * args.decay
    print("AMG End!")
    print(s)
    print(t)
    return s, t

def train(node_feats, node_types, adjs, pos_train, neg_train, pos_val, neg_val, model_s, model_t, optimizer_w, optimizer_a, eps):
    idxes_seq_s, idxes_res_s = model_s.sample(eps)
    idxes_seq_t, idxes_res_t = model_t.sample(eps)

    # 2.对w和a交替更新训练
    optimizer_w.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t)
    loss_w = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                          F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss_w.backward()
    optimizer_w.step()

    optimizer_a.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t)
    loss_a = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)) + \
                          F.logsigmoid(- torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)))
    loss_a.backward()
    optimizer_a.step()

    return loss_w.item(), loss_a.item()