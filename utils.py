import numpy as np
from pathlib import Path
import scipy.sparse as sp
from typing import *
import json


def load_data(data_set) -> Tuple:
    """
    :param data_set: 数据集名称
    :return: 邻接矩阵和ripple distance矩阵
    """
    adj_path = Path(__file__).parent / 'adjacentMatrix' / (data_set + '.npz')
    ripple_path = Path(__file__).parent / 'rippleSimilarityMatrix' / (data_set + '.npz')
    adj_mat = sp.load_npz(adj_path)
    ripple_mat = sp.load_npz(ripple_path)
    # adj_mat=normalize(adj_mat)
    adj_mat, ripple_mat = np.array(adj_mat.todense()), np.array(ripple_mat.todense())
    adj_mat = normalize_np(adj_mat)
    #adj_mat = helper(adj_mat)
    ripple_mat = helper(ripple_mat)
    return adj_mat, ripple_mat


def normalize_sp(mx: sp.coo_matrix) -> sp.coo_matrix:
    rows_sum = np.array(mx.sum(1)).astype('float')  # 对每一行求和
    rows_inv = np.power(rows_sum, -1).flatten()  # 求倒数
    rows_inv[np.isinf(rows_inv)] = 0  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    rows_inv = np.sqrt(rows_inv)
    rows_mat_inv = sp.diags(rows_inv)  # 构建对角元素为r_inv的对角矩阵
    cols_sum = np.array(mx.sum(0)).astype('float')
    cols_inv = np.power(cols_sum, -1).flatten()
    cols_inv[np.isinf(cols_inv)] = 0
    cols_inv = np.sqrt(cols_inv)
    cols_mat_inv = sp.diags(cols_inv)
    mx = rows_mat_inv.dot(mx).dot(cols_mat_inv)
    return mx


def normalize_np(mx: np.ndarray) -> np.ndarray:
    rows_sum = np.array(mx.sum(1)).astype('float')  # 对每一行求和
    rows_inv = np.power(rows_sum, -1).flatten()  # 求倒数
    rows_inv[np.isinf(rows_inv)] = 0  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    rows_inv = np.sqrt(rows_inv)
    rows_mat_inv = np.diag(rows_inv)  # 构建对角元素为r_inv的对角矩阵
    cols_sum = np.array(mx.sum(0)).astype('float')
    cols_inv = np.power(cols_sum, -1).flatten()
    cols_inv[np.isinf(cols_inv)] = 0
    cols_inv = np.sqrt(cols_inv)
    cols_mat_inv = np.diag(cols_inv)
    mx = rows_mat_inv.dot(mx).dot(cols_mat_inv)
    return mx


def helper(mat: np.ndarray):
    max_vals = np.max(mat, axis=1)
    max_vals = np.power(max_vals, -1)
    max_vals[np.isinf(max_vals)] = 0
    return np.diag(max_vals).dot(mat)


def load_params(dataset):
    path = Path(__file__).parent / 'params' / (dataset + '.json')
    f = open(path, 'r')
    params = json.load(f)
    return params


if __name__ == '__main__':
    mat1, mat2 = load_data('barbell')
    mat1 = normalize_np(mat1)
    print(mat1)
