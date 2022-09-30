import numpy as np
import BA

m = 5
n = 20
N = 100000

#np.random.seed(10086)

edge_list = BA.BA_graph(5, m, n - 5)
match1 = []
match2 = []
for edge in edge_list:
    k = np.random.randint(0, 2)
    match1.append(edge[k])
    match2.append(edge[1 - k])
m = len(edge_list)
print("edge number:", m)
'''
match1 = np.random.randint(0, n, size=m)
match2 = np.random.randint(0, n, size=m)
'''
w1 = abs(np.random.normal(size=(m, m, 1)))
w2 = 2 * abs(np.random.normal(size=(m, m, 1)))
w0 = np.concatenate((w1, w2), axis=2)


def feed(match):
    maps = {}
    for i in range(m):
        if not match[i] in maps:
            maps[match[i]] = []
        maps[match[i]].append(i)
    return maps


def aggregate_func(lists):
    return np.mean(lists)
    return 1 / (1 + np.exp(-np.sum(lists)))


def cal_outcome(w, match):
    maps = feed(match)
    res = 0.0
    for i in range(m):
        j = match[i]
        res += w[i][i] + aggregate_func(list(map(lambda x: w[i][x], maps[j])))
    return res


def estimator1(w, match1, match2, p=0.5):
    prob1 = np.zeros(n)
    prob2 = np.zeros(n)
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_match = np.zeros(m, dtype=int)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                if np.random.random() > p:
                    for k in groups[i][j]:
                        real_match[k] = 1
                else:
                    for k in groups[i][j]:
                        real_match[k] = 0
    res1 = 0.0
    res2 = 0.0
    map1 = feed(match1)
    map2 = feed(match2)
    map3 = feed([match2[i] if real_match[i] else match1[i] for i in range(m)])
    for j in range(n):
        if j in map3 and j in map1:
            if sorted(map3[j]) == sorted(map1[j]):
                prob1[j] += 1
                factor = 1.0
                group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                factor *= p**len(group1)
                if j in map2:
                    group2 = set(
                        [match1[i] for i in map2[j] if match1[i] != j])
                    factor *= (1 - p)**len(group2)
                for i in map3[j]:
                    temp = w[i][i][real_match[i]] + aggregate_func(
                        list(map(lambda x: w[i][x][real_match[x]], map3[j])))
                    res1 += temp / factor
                prob2[j] = factor
        if j in map3 and j in map2:
            if sorted(map3[j]) == sorted(map2[j]):
                factor = 1.0
                group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                factor *= (1 - p)**len(group2)
                if j in map1:
                    group1 = set(
                        [match2[i] for i in map1[j] if match2[i] != j])
                    factor *= p**len(group1)
                for i in map3[j]:
                    temp = w[i][i][real_match[i]] + aggregate_func(
                        list(map(lambda x: w[i][x][real_match[x]], map3[j])))
                    res2 += temp / factor
    return res1 / m, res2 / m, prob1, prob2


def estimator2(w, match1, match2, p=0.5):
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_match = np.zeros(m, dtype=int)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                if np.random.random() > p:
                    for k in groups[i][j]:
                        real_match[k] = 1
                else:
                    for k in groups[i][j]:
                        real_match[k] = 0
    res1 = 0.0
    res2 = 0.0
    weight1 = 0.0
    weight2 = 0.0
    map1 = feed(match1)
    map2 = feed(match2)
    map3 = feed([match2[i] if real_match[i] else match1[i] for i in range(m)])
    for j in range(n):
        if j in map3 and j in map1:
            if sorted(map3[j]) == sorted(map1[j]):
                factor = 1.0
                group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                factor *= p**len(group1)
                if j in map2:
                    group2 = set(
                        [match1[i] for i in map2[j] if match1[i] != j])
                    factor *= (1 - p)**len(group2)
                for i in map3[j]:
                    temp = w[i][i][real_match[i]] + aggregate_func(
                        list(map(lambda x: w[i][x][real_match[x]], map3[j])))
                    res1 += temp / factor
                    weight1 += 1.0 / factor
        if j in map3 and j in map2:
            if sorted(map3[j]) == sorted(map2[j]):
                factor = 1.0
                group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                factor *= (1 - p)**len(group2)
                if j in map1:
                    group1 = set(
                        [match2[i] for i in map1[j] if match2[i] != j])
                    factor *= p**len(group1)
                for i in map3[j]:
                    temp = w[i][i][real_match[i]] + aggregate_func(
                        list(map(lambda x: w[i][x][real_match[x]], map3[j])))
                    res2 += temp / factor
                    weight2 += 1.0 / factor
    if weight1:
        res1 /= weight1
    if weight2:
        res2 /= weight2
    return res1, res2, weight1 / m, weight2 / m


def estimator3(w, match1, match2, p=0.5):
    real_match = np.array(
        [1 if np.random.random() > p else 0 for i in range(m)], dtype=int)
    maps = feed([match2[i] if real_match[i] else match1[i] for i in range(m)])
    res1 = 0.0
    res2 = 0.0
    for i in range(m):
        j = match2[i] if real_match[i] else match1[i]
        temp = w[i][i][real_match[i]] + aggregate_func(
            list(map(lambda x: w[i][x][real_match[x]], maps[j])))
        if real_match[i]:
            res2 += temp
        else:
            res1 += temp
    if m - np.sum(real_match):
        res1 /= m - np.sum(real_match)
    if np.sum(real_match):
        res2 /= np.sum(real_match)
    return res1, res2


def estimator4(w, match1, match2, p=0.5):
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_match = np.zeros(m, dtype=int)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                if np.random.random() > p:
                    for k in groups[i][j]:
                        real_match[k] = 1
                else:
                    for k in groups[i][j]:
                        real_match[k] = 0
    maps = feed([match2[i] if real_match[i] else match1[i] for i in range(m)])
    res1 = 0.0
    res2 = 0.0
    for i in range(m):
        j = match2[i] if real_match[i] else match1[i]
        temp = w[i][i][real_match[i]] + aggregate_func(
            list(map(lambda x: w[i][x][real_match[x]], maps[j])))
        if real_match[i]:
            res2 += temp
        else:
            res1 += temp
    if m - np.sum(real_match):
        res1 /= m - np.sum(real_match)
    if np.sum(real_match):
        res2 /= np.sum(real_match)
    return res1, res2


if __name__ == "__main__":
    outcome1 = cal_outcome(w1, match1)
    outcome2 = cal_outcome(w2, match2)
    np.random.seed()
    '''
    test1 = [1, 1, 2, 1, 3]
    test2 = [1, 1, 1, 2, 2]
    test_w = abs(np.random.normal(size=(5, 5)))
    estimator2(test_w, test1, test2)
    '''
    print("real average outcome:", outcome1 / m, outcome2 / m)
    print("---------------------------------------------------------")

    res1 = []
    res2 = []
    prob1 = np.zeros(n)
    prob2 = np.zeros(n)
    for k in range(N):
        temp1, temp2, temp3, temp4 = estimator1(w0, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
        prob1 += temp3
        prob2 = np.maximum(prob2, temp4)
    print("method 1 expectation:", np.mean(res1), np.mean(res2))
    print("method 1 variance:", np.std(res1, ddof=1), np.std(res2, ddof=1))
    #for i in range(m):
    #print("node ", i, prob1[i] / N, prob2[i])
    print("---------------------------------------------------------")
    #'''
    res1 = []
    res2 = []
    weight1 = []
    weight2 = []
    for k in range(N):
        temp1, temp2, temp3, temp4 = estimator2(w0, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
        weight1.append(temp3)
        weight2.append(temp4)
    print("method 2 expectation:", np.mean(res1), np.mean(res2),
          np.mean(weight1), np.mean(weight2))
    print("method 2 variance:", np.std(res1, ddof=1), np.std(res2, ddof=1),
          np.std(weight1, ddof=1), np.std(weight2, ddof=1))
    print("---------------------------------------------------------")

    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator3(w0, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 3 expectation:", np.mean(res1), np.mean(res2))
    print("method 3 variance:", np.var(res1, ddof=1), np.var(res2, ddof=1))
    print("---------------------------------------------------------")
    #'''
    #'''
    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator4(w0, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 4 expectation:", np.mean(res1), np.mean(res2))
    print("method 4 variance:", np.var(res1, ddof=1), np.var(res2, ddof=1))
    print("---------------------------------------------------------")
    #'''
