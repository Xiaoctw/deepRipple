import numpy as np
from pathlib import Path

dic1 = {}

dataset = 'barbell'
# 这里使用的数据集为'usa-airports','brazil-airports','europe-airports'

file_path = Path(__file__).parent.parent / 'rawData' / (dataset + '.edgelist')
write_path = Path(__file__).parent / (dataset + '.edgelist')
map_path = Path(__file__).parent / (dataset + '.txt')

if __name__ == '__main__':
    idx = 0
    f = open(write_path, 'w')
    f1 = open(map_path, 'w')
    for line in open(file_path).readlines():
        list1 = line.strip().split()
        val1, val2 = int(list1[0]), int(list1[1])
        if val1 not in dic1:
            dic1[val1] = idx
            idx += 1
        if val2 not in dic1:
            dic1[val2] = idx
            idx += 1
        line1 = str(dic1[val1]) + ' ' + str(dic1[val2]) + '\n'
        f.write(line1)
    for key in dic1:
        line = str(dic1[key]) + ' ' + str(key) + '\n'
        f1.write(line)
        # rows.append(int(list1[0]))
        # cols.append(int(list1[1]))
        # data.append(1)
        # rows.append(int(list1[1]))
        # cols.append(int(list1[0]))
        # max_val=max(max_val,int(list1[1]),int(list1[0]))
        # data.append(1)
