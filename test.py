import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def LCC(t, k):
    if k > 1:
        return t/comb(k, 2)
    return 0


with open("sx-mathoverflow-a2q.txt", "r") as f:
    N = 90000
    edges = {}
    data = f.readlines()
    tri = {}
    counts = []
    sum = 0
    total_edges = 0
    CC = 0.0
    count = 0
    for line in data:
        count += 1
        ids = line.split(' ')
        if ids[0] == ids[1]:
            continue
        edges.setdefault(ids[0], {})
        edges.setdefault(ids[1], {})
        tri.setdefault(ids[0], 0)
        tri.setdefault(ids[1], 0)
        if not ids[1] in edges[ids[0]]:
            CC -= LCC(tri[ids[0]], len(edges[ids[0]])) + \
                LCC(tri[ids[1]], len(edges[ids[1]]))
            for key in edges[ids[1]].keys():
                # if ids[0] in edges.setdefault(key, {}):
                if key in edges[ids[0]].keys():
                    CC -= LCC(tri[key], len(edges[key]))
                    tri[key] += 1
                    tri[ids[0]] += 1
                    tri[ids[1]] += 1
                    CC += LCC(tri[key], len(edges[key]))
            CC += LCC(tri[ids[0]], len(edges[ids[0]])) + \
                LCC(tri[ids[1]], len(edges[ids[1]]))
        edges[ids[0]].setdefault(ids[1], 0)
        edges[ids[1]].setdefault(ids[0], 0)
        edges[ids[0]][ids[1]] += 1
        edges[ids[1]][ids[0]] += 1
        counts.append(CC)

x = range(len(counts))
plt.plot(x, np.array(counts)/len(edges))
plt.show()

print(counts[-1])

threshold = 0
large_tri = 0
for i in edges.keys():
    for j in edges[i].keys():
        if edges[i][j] > threshold:
            for k in edges[j].keys():
                if edges[j][k] > threshold and k in edges[i].keys() and edges[i][k] > threshold:
                    large_tri += 1
print(large_tri/6)
