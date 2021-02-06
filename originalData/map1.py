import numpy as np
from pathlib import Path
from collections import defaultdict
import scipy.sparse as sp

dic1 = {}

dataset = 'cora'
# 这里使用的数据集为'usa-airports','brazil-airports','europe-airports'

file_path = Path(__file__).parent.parent / 'rawData' / (dataset + '.npz')
write_path = Path(__file__).parent / (dataset + '.edgelist')
map_path = Path(__file__).parent / (dataset + '.txt')

if __name__ == '__main__':
    idx = 0
    mat1 = sp.load_npz(file_path).tocoo()
    rows=mat1.row
    cols=mat1.col
    f = open(write_path, 'w')
    f1 = open(map_path, 'w')
    dic1 = defaultdict(lambda x: x)
    for i in range(len(rows)):
        line = str(rows[i]) + ' ' + str(cols[i]) + '\n'
        f.write(line)
    for i in range(len(cols)):
        line = str(i) + ' ' + str(i) + '\n'
        f1.write(line)

        # rows.append(int(list1[0]))
        # cols.append(int(list1[1]))
        # data.append(1)
        # rows.append(int(list1[1]))
        # cols.append(int(list1[0]))
        # max_val=max(max_val,int(list1[1]),int(list1[0]))
        # data.append(1)
