import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def generate_data(seed, num_drug, data_path):
    np.random.seed(seed)
    dp = pd.read_csv((data_path), encoding='utf-8', delimiter=',',
                         names=['did', 'pid', 'rating'])
    dp_pos = dp[dp['rating'] == 1].to_numpy()[:, :2]
    dp_pos[:, 1] += num_drug
    neg_pos = dp[dp['rating'] == 0].to_numpy()[:, :2]
    neg_pos[:, 1] += num_drug
    assert (dp_pos.shape[0] + neg_pos.shape[0] == dp.shape[0])

    kf = KFold(n_splits=5, shuffle=False)
    # * negative pairs
    indices_neg = np.arange(neg_pos.shape[0])
    np.random.shuffle(indices_neg)
    indices_neg = indices_neg[:dp_pos.shape[0]]
    neg_pos = neg_pos[indices_neg]
    neg_train_fold = []
    neg_val_fold = []
    neg_test_fold = []

    for fold, (train_indices, test_indices) in enumerate(kf.split(neg_pos)):
        neg_train = neg_pos[train_indices]
        neg_test = neg_pos[test_indices]
        val_size = len(neg_test)
        neg_val = neg_train[:val_size]
        neg_train = neg_train[val_size:]

        neg_train_fold.append(neg_train)
        neg_val_fold.append(neg_val)
        neg_test_fold.append(neg_test)
    np.savez("./preprocessed/neg_pairs_offset", train=neg_train_fold, val=neg_val_fold, test=neg_test_fold, allow_pickle=False)

    # * positive pairs
    indices = np.arange(dp_pos.shape[0])
    np.random.shuffle(indices)
    dp_pos = dp_pos[indices]
    pos_train_fold = []
    pos_val_fold = []
    pos_test_fold = []
    for fold, (train_indices, test_indices) in enumerate(kf.split(dp_pos)):
        pos_train = dp_pos[train_indices]
        pos_test = dp_pos[test_indices]
        val_size = len(pos_test)
        pos_val = pos_train[:val_size]
        pos_train = pos_train[val_size:]

        pos_train_fold.append(pos_train)
        pos_val_fold.append(pos_val)
        pos_test_fold.append(pos_test)
    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold