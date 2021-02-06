import os
import sys

names = ['brazil', 'europe', 'usa']
if __name__ == "__main__":
    name = names[1]
    origin = "originalData"
    emb = "embedding_results"
    file1 = name + "-airports.txt"
    file2 = name + "-airports_outVec.txt"
    reflect = os.path.join(origin, file1)
    out = os.path.join(emb, file2)
    out2 = os.path.join(emb, name + ".emb")
    r = dict()
    with open(reflect, 'r') as file:
        lines = file.readlines()
        for line in lines:
            items = line.strip().split()
            r[int(items[0])] = int(items[1])
    with open(out, 'r') as file:
        lines = file.readlines()
        length = len(lines[0].strip().split())
        with open(out2, 'w') as file2:
            file2.write("%d\t%d\n" % (len(lines), length))
            for i in range(len(lines)):
                file2.write("%d %s" % (r[i], lines[i]))
