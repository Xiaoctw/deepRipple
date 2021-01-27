from pathlib import Path
import scipy.sparse as sp
import numpy as np

data_set = 'usa-airports'

file_path = Path(__file__).parent.parent / 'originalData' / (data_set + '.edgelist')
save_path = Path(__file__).parent.parent / 'adjacentMatrix' / (data_set + '.npz')

if __name__ == '__main__':
    rows, cols, data = [], [], []
    max_val = 0
    for line in open(file_path).readlines():
        list1 = line.strip().split()
        rows.append(int(list1[0]))
        cols.append(int(list1[1]))
        data.append(1)
        rows.append(int(list1[1]))
        cols.append(int(list1[0]))
        max_val = max(max_val, int(list1[1]), int(list1[0]))
        data.append(1)
    # print('shape:{}'.format(max_val))
    coo = sp.coo_matrix((data, (rows, cols)), shape=(max_val + 1, max_val + 1))
    sp.save_npz(save_path, coo)
