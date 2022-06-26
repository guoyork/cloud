import numpy as np
import scipy
import cvxpy as cp
from collections import Counter
from scipy.special import comb, perm

N = 10000
m = 100
n = 10
epsilon = 1e-7
interval = 1e-1
np.random.seed(10)


def vec2mat(vec, value=np.ones(m)):
    res = np.zeros((m, n))
    for i in range(m):
        res[i][vec[i]] += value[i]
    return res


def set_budget(match1, match0, costs):
    temp1 = vec2mat(match1)
    temp0 = vec2mat(match0)
    cost1 = np.sum(np.multiply(temp1, costs), axis=0)
    cost0 = np.sum(np.multiply(temp0, costs), axis=0)
    budget = np.maximum(cost1, cost0)+.1
    return budget


def settle(ad_id, budget, revenue, cost):
    if budget[ad_id] >= cost:
        budget[ad_id] -= cost
        return budget, revenue
    else:
        return budget, 0


def cal_outcome(w, costs, budget, match):
    res = 0.0
    used_budget = budget.copy()
    for i in range(m):
        used_budget, temp = settle(match[i], used_budget, w[i], costs[i][match[i]])
        res += temp
    return res


def estimator1(outcomes, costs, budget, match1, match0, p):
    p = np.append(p, np.maximum(1-np.sum(p, axis=1), 0).reshape(-1, 1), axis=1)
    real_match = [np.random.choice(range(0, n+1), p=p[i]) for i in range(m)]
    res1 = 0.0
    res0 = 0.0
    used_budget = budget.copy()
    for i in range(m):
        if real_match[i] == n:
            continue
        used_budget, temp = settle(real_match[i], used_budget, outcomes[i][real_match[i]], costs[i][real_match[i]])
        if real_match[i] == match1[i]:
            res1 += temp/p[i][match1[i]]
        if real_match[i] == match0[i]:
            res0 += temp/p[i][match0[i]]
    return res1, res0


def optimize(outcomes, budget, intervals):
    x = cp.Variable((m, n))
    objective = cp.Minimize(cp.sum(cp.multiply(outcomes**2, cp.exp(-x))))
    constraints = [cp.sum(cp.exp(x), axis=1) <= 1, cp.sum(cp.multiply(costs, cp.exp(x)), axis=0) <= budget-intervals]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver='ECOS', max_iters=200)
    return np.array(np.exp(x.value))


if __name__ == "__main__":
    match0 = np.random.randint(0, n, size=m)
    match1 = np.random.randint(0, n, size=m)
    for i in range(m):
        if match0[i] == match1[i]:
            match0[i] = (match1[i]+1) % n
    costs = np.random.random((m, n))

    w0 = np.concatenate((2 * abs(np.random.normal(size=(m//2, 1))), 4*abs(np.random.normal(size=(m//2, 1)))), axis=0)
    w1 = np.concatenate((abs(np.random.normal(size=(m//2, 1))), 2*abs(np.random.normal(size=(m//2, 1)))), axis=0)
    w = np.concatenate((w0, w1), axis=1)
    outcomes = vec2mat(match1, w1)+vec2mat(match0, w0)
    budget = set_budget(match1, match0, costs)
    print(cal_outcome(w0, costs, budget, match0)-cal_outcome(w1, costs, budget, match1))

    res = []

    abtests = (vec2mat(match0)+vec2mat(match1))/2
    res1 = []
    res0 = []
    for i in range(N):
        temp1, temp0 = estimator1(outcomes, costs, budget, match1, match0, abtests)
        res1.append(temp1)
        res0.append(temp0)
    print(np.mean(res0)-np.mean(res1))
    res.append(np.std(np.array(res1)-np.array(res0)))
    print(np.std(np.array(res1)-np.array(res0)))
    print("----------------------------------------")
    '''
    res1 = []
    res0 = []
    opt = optimize(outcomes, budget, 0)
    for i in range(N):
        temp1, temp0 = estimator1(outcomes, costs, budget, match1, match0, opt)
        res1.append(temp1)
        res0.append(temp0)
    print(np.mean(res0)-np.mean(res1))
    print(np.std(np.array(res1)-np.array(res0)))
    print("----------------------------------------")
    '''
    for k in range(30):
        res1 = []
        res0 = []
        opt = optimize(outcomes, budget, k*interval)
        for i in range(N):
            temp1, temp0 = estimator1(outcomes, costs, budget, match1, match0, opt)
            res1.append(temp1)
            res0.append(temp0)
        print(np.mean(res0)-np.mean(res1))
        print(np.std(np.array(res1)-np.array(res0)))
        res.append(np.std(np.array(res1)-np.array(res0)))
        print("episode:"+str(k)+"----------------------------------------")
    np.savetxt('2.txt', np.asarray(res))
