from pathlib import Path
import scipy.sparse as sp
import numpy as np

data_set = 'usa-airports'

load_path = Path(__file__).parent.parent / 'rippleDistanceMatrix' / (data_set + '.npz')
save_path = Path(__file__).parent / (data_set + '.npz')

# 通过
if __name__ == '__main__':
    print('读取数据')
    data = sp.load_npz(load_path)
    val_list = np.array(data.data)
    row, col = data.row, data.col
    print('数据个数:{}'.format(len(col)))
    # val_list = np.power(val_list, -1).flatten()
    val_list1 = []
    for val in val_list:
        val_list1.append(9 - val)
    # val_list[np.isinf(val_list)] = 1
    new_data = sp.coo_matrix((val_list1, (row, col)), shape=data.shape)
    sp.save_npz(save_path, new_data)
    print('构造完成')
