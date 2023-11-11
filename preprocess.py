import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os
import pickle

def normalize_sym(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_row(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_Luo(prefix):
    dp = pd.read_csv(os.path.join(prefix, "drug_target.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'pid', 'rating']).reset_index(drop=True)
    dd = pd.read_csv(os.path.join(prefix, "drug_drug.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'd2', 'weight']).reset_index(drop=True)
    simdd = pd.read_csv(os.path.join(prefix, "sim_drugs.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'd2', 'weight']).reset_index(drop=True)
    pp = pd.read_csv(os.path.join(prefix, "pro_pro.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'p2', 'weight']).reset_index(drop=True)
    simpp = pd.read_csv(os.path.join(prefix, "sim_proteins.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'p2', 'weight']).reset_index(drop=True)
    de = pd.read_csv(os.path.join(prefix, "drug_dis.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'dis', 'weight']).reset_index(drop=True)
    pe = pd.read_csv(os.path.join(prefix, "protein_dis.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'dis', 'weight']).reset_index(drop=True)
    ds = pd.read_csv(os.path.join(prefix, "drug_se.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'se', 'weight']).reset_index(drop=True)
    np.random.seed(1)

    offsets = {'drug': 708, 'protein': 708 + 1512}
    offsets['disease'] = offsets['protein'] + 5603
    offsets['sideeffect'] = offsets['disease'] + 4192
    print(offsets['sideeffect'])
    # * node types
    node_types = np.zeros((offsets['sideeffect'],), dtype=np.int32)
    node_types[offsets['drug']:offsets['protein']] = 1
    node_types[offsets['protein']:offsets['disease']] = 2
    node_types[offsets['disease']:] = 3

    np.save("./preprocessed/node_types", node_types)

    #* positive pairs
    dp_pos = dp[dp['rating'] == 1].to_numpy()[:, :2]
    #* adjs with offset
    adjs_offset = {}

    # drug-protein
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[dp_pos[:, 0], dp_pos[:, 1] + offsets['drug']] = 1
    adjs_offset['0'] = sp.coo_matrix(adj_offset)
    print(len(dp_pos))

    # drug-disease
    de_npy = de.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[de_npy[:, 0], de_npy[:, 1] + offsets['protein']] = 1
    adjs_offset['1'] = sp.coo_matrix(adj_offset)
    print(len(de_npy))
    ed_matrix = np.zeros((5603, 708), dtype=int)
    ed_matrix[de_npy[:, 1], de_npy[:, 0]] = 1

    # protein-disease
    pe_npy = pe.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[pe_npy[:, 0] + offsets['drug'], pe_npy[:, 1] + offsets['protein']] = 1
    adjs_offset['2'] = sp.coo_matrix(adj_offset)
    print(len(pe_npy))

    # drug-sideeffect
    ds_npy = ds.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ds_npy[:, 0], ds_npy[:, 1] + offsets['disease']] = 1
    adjs_offset['3'] = sp.coo_matrix(adj_offset)
    print(len(ds_npy))
    sd_matrix = np.zeros((4192, 708), dtype=int)
    sd_matrix[ds_npy[:, 1], ds_npy[:, 0]] = 1

    # drug-drug
    dd_npy = dd.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[dd_npy[:, 0], dd_npy[:, 1]] = 1
    adjs_offset['4'] = sp.coo_matrix(adj_offset)
    print(len(dd_npy))
    dd_matrix = np.zeros((708, 708), dtype=int)
    dd_matrix[dd_npy[:, 0], dd_npy[:, 1]] = 1

    # simdd
    simdd_npy = simdd.to_numpy(int)[:, :2]
    dd_score = simdd['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(simdd_npy[:, 0], simdd_npy[:, 1], dd_score):
        adj_offset[i, j] = k
    adjs_offset['5'] = sp.coo_matrix(adj_offset)
    print(len(simdd_npy))

    # protein-protein
    pp_npy = pp.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[pp_npy[:, 0] + offsets['drug'], pp_npy[:, 1] + offsets['drug']] = 1
    adjs_offset['6'] = sp.coo_matrix(adj_offset)
    print(len(pp_npy))
    pp_matrix = np.zeros((1512, 1512), dtype=int)
    pp_matrix[pp_npy[:, 0], pp_npy[:, 1]] = 1

    # simpp
    simpp_npy = simpp.to_numpy(int)[:, :2]
    pp_score = simpp['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(simpp_npy[:, 0] + offsets['drug'], simpp_npy[:, 1] + offsets['drug'], pp_score):
        adj_offset[i, j] = k
    adjs_offset['7'] = sp.coo_matrix(adj_offset)
    print(len(simpp_npy))

    f2 = open("./preprocessed/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()
    np.savez('./preprocessed/combined_matrices.npz', ed_matrix=ed_matrix, sd_matrix=sd_matrix)

def preprocess_Zheng(prefix):
    dp = pd.read_csv(os.path.join(prefix, "drug_target.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'pid', 'rating'])
    chemical_dd = pd.read_csv(os.path.join(prefix, "drug_chemical_substructures.dat"), encoding='utf-8', delimiter=',',
                              names=['d1', 'd2', 'weight'])
    ds = pd.read_csv(os.path.join(prefix, "drug_sideeffects.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'dis', 'weight'])
    stituent_dd = pd.read_csv(os.path.join(prefix, "drug_sub_stituent.dat"), encoding='utf-8', delimiter=',',
                              names=['d1', 'd2', 'weight'])
    simdd = pd.read_csv(os.path.join(prefix, "drug_chemical_sim.dat"), encoding='utf-8', delimiter=',',
                        names=['d1', 'd2', 'weight'])
    pp = pd.read_csv(os.path.join(prefix, "target_GO.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'p2', 'weight'])
    simpp = pd.read_csv(os.path.join(prefix, "target_GO_sim.dat"), encoding='utf-8', delimiter=',',
                        names=['p1', 'p2', 'weight'])

    np.random.seed(1)
    offsets = {'drug': 1094, 'protein': 1094 + 1556}
    offsets['chemical'] = offsets['protein'] + 881
    offsets['sideeffict'] = offsets['chemical'] + 4063
    offsets['stituent'] = offsets['sideeffict'] + 738
    offsets['go'] = offsets['stituent'] + 4098
    # * node types
    node_types = np.zeros((offsets['go'],), dtype=np.int32)
    node_types[offsets['drug']:offsets['protein']] = 1
    node_types[offsets['protein']:offsets['chemical']] = 2
    node_types[offsets['chemical']:offsets['sideeffict']] = 3
    node_types[offsets['sideeffict']:offsets['stituent']] = 4
    node_types[offsets['stituent']:] = 5

    if not os.path.exists("./preprocessed/Drug/node_types.npy"):
        np.save("./preprocessed/node_types", node_types)

    dp_pos = dp[dp['rating'] == 1].to_numpy()[:, :2]
    # * adjs with offset
    adjs_offset = {}
    # dp
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[dp_pos[:, 0], dp_pos[:, 1] + offsets['drug']] = 1
    adjs_offset['0'] = sp.coo_matrix(adj_offset)
    print(len(dp_pos))

    # chemical_dd
    chemical_dd_npy = chemical_dd.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[chemical_dd_npy[:, 0], chemical_dd_npy[:, 1] + offsets['protein']] = 1
    adjs_offset['2'] = sp.coo_matrix(adj_offset)
    print(len(chemical_dd_npy))
    cd_matrix = np.zeros((881, 1094), dtype=int)
    cd_matrix[chemical_dd_npy[:, 1], chemical_dd_npy[:, 0]] = 1

    # ds
    ds_npy = ds.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ds_npy[:, 0], ds_npy[:, 1] + offsets['chemical']] = 1
    adjs_offset['1'] = sp.coo_matrix(adj_offset)
    print(len(ds_npy))
    sd_matrix = np.zeros((4063, 1094), dtype=int)
    sd_matrix[ds_npy[:, 1], ds_npy[:, 0]] = 1

    # stituent_dd
    stituent_dd_npy = stituent_dd.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[stituent_dd_npy[:, 0], stituent_dd_npy[:, 1] + offsets['sideeffict']] = 1
    adjs_offset['4'] = sp.coo_matrix(adj_offset)
    print(len(stituent_dd_npy))
    stid_matrix = np.zeros((738, 1094), dtype=int)
    stid_matrix[stituent_dd_npy[:, 1], stituent_dd_npy[:, 0]] = 1

    # go_p
    gop_npy = pp.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[gop_npy[:, 0] + offsets['drug'], gop_npy[:, 1] + offsets['drug']] = 1
    adjs_offset['3'] = sp.coo_matrix(adj_offset)
    print(len(gop_npy))
    gop_matrix = np.zeros((4098, 1556), dtype=int)
    gop_matrix[gop_npy[:, 1], gop_npy[:, 0]] = 1

    # simdd
    simdd_npy = simdd.to_numpy(int)[:, :2]
    dd_score = simdd['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(simdd_npy[:, 0], simdd_npy[:, 1], dd_score):
        adj_offset[i, j] = k
    adjs_offset['6'] = sp.coo_matrix(adj_offset)
    print(len(simdd_npy))

    # simpp
    simpp_npy = simpp.to_numpy(int)[:, :2]
    pp_score = simpp['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(simpp_npy[:, 0] + offsets['drug'], simpp_npy[:, 1] + offsets['drug'], pp_score):
        adj_offset[i, j] = k
    adjs_offset['5'] = sp.coo_matrix(adj_offset)
    print(len(simpp_npy))

    f2 = open("./preprocessed/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()
    np.savez('./preprocessed/combined_matrices.npz', cd_matrix=cd_matrix,
             sd_matrix=sd_matrix, stid_matrix=stid_matrix, gop_matrix=gop_matrix)

if __name__ == '__main__':
    preprocess_Luo("./data/Luo")
    # preprocess_Zheng("./data/Zheng")