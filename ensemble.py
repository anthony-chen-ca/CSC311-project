from utils import *
from knn import *
import numpy as np
import os
import csv
import sys
np.set_printoptions(threshold=sys.maxsize) # To print full matrix/array instead of truncated ones.

def main():
    userid_dim = 542
    questionid_dim = 1774

    root_dir = os.path.abspath("./data")
    train_matrix = load_train_sparse(root_dir).toarray()
    val_data = load_valid_csv(root_dir)
    test_data = load_public_test_csv(root_dir)

    train_size = train_matrix.shape[0]
    train_indices = np.arange(train_size)
    def create_bootstrap_sample(matrix):
        sample_indices = np.random.choice(train_indices, train_size, replace=True)
        return matrix[sample_indices]

    train_matrix_1 = create_bootstrap_sample(train_matrix)
    train_matrix_2 = create_bootstrap_sample(train_matrix)
    train_matrix_3 = create_bootstrap_sample(train_matrix)


    k_val = [1, 6, 11, 16, 21, 26]
    knn_best_k_1 = k_val[0]
    knn_best_acc_1 = 0
    for k in k_val:
        acc_temp = knn_impute_by_user(train_matrix_1, val_data, k)
        if (acc_temp > knn_best_acc_1):
            knn_best_acc_1 = acc_temp
            knn_best_k_1 = k

    knn_best_k_2 = k_val[0]
    knn_best_acc_2 = 0
    for k in k_val:
        acc_temp = knn_impute_by_user(train_matrix_2, val_data, k)
        if (acc_temp > knn_best_acc_2):
            knn_best_acc_2 = acc_temp
            knn_best_k_2 = k


    knn_best_k_3 = k_val[0]
    knn_best_acc_3 = 0
    for k in k_val:
        acc_temp = knn_impute_by_user(train_matrix_3, val_data, k)
        if (acc_temp > knn_best_acc_3):
            knn_best_acc_3 = acc_temp
            knn_best_k_3 = k

    nbrs = KNNImputer(n_neighbors=knn_best_k_1)
    pred_knn_1 = nbrs.fit_transform(train_matrix_1) 

    nbrs = KNNImputer(n_neighbors=knn_best_k_2)
    pred_knn_2 = nbrs.fit_transform(train_matrix_2) 

    nbrs = KNNImputer(n_neighbors=knn_best_k_3)
    pred_knn_3 = nbrs.fit_transform(train_matrix_3) 

    pred_final = (pred_knn_1 + pred_knn_2 + pred_knn_3) / 3

    acc_valid_final = sparse_matrix_evaluate(val_data, pred_final) 
    acc_test_final = sparse_matrix_evaluate(test_data, pred_final)
    print("Final validation accuracy: " + str(acc_valid_final))
    print("Final test accuracy: " + str(acc_test_final))


if __name__ == "__main__":
    main()