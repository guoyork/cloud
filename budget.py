import numpy as np
from regex import I
import scipy
import cvxpy as cp
from collections import Counter
from scipy.special import comb, perm

N = 10000
m = 5
n = 5
epsilon = 1e-7
interval = 2e-1
#np.random.seed(100000)


def vec2mat(vec, value):
    res = np.zeros((len(vec), n))
    for i in range(len(vec)):
        res[i][vec[i]] += value[i]
    return res


def set_budget(match1, match0, costs):
    temp1 = vec2mat(match1, np.ones(len(match1)))
    temp0 = vec2mat(match0, np.ones(len(match0)))
    cost1 = np.sum(np.multiply(temp1, costs), axis=0)
    cost0 = np.sum(np.multiply(temp0, costs), axis=0)
    budget = np.maximum(cost1, cost0) + .1
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
    p = np.append(p, np.maximum(1 - np.sum(p, axis=1), 0).reshape(-1, 1), axis=1)
    real_match = [np.random.choice(range(0, n + 1), p=p[i]) for i in range(m)]
    res1 = 0.0
    res0 = 0.0
    used_budget = budget.copy()
    for i in range(m):
        if real_match[i] == n:
            continue
        used_budget, temp = settle(real_match[i], used_budget, outcomes[i][real_match[i]], costs[i][real_match[i]])
        if real_match[i] == match1[i]:
            res1 += temp / p[i][match1[i]]
        if real_match[i] == match0[i]:
            res0 += temp / p[i][match0[i]]
    return res1, res0


def estimator2(outcomes, costs, budget, match1, match0, p):
    p = np.append(p, np.maximum(1 - np.sum(p, axis=1), 0).reshape(-1, 1), axis=1)
    real_match = [np.random.choice(range(0, n + 1), p=p[i]) for i in range(m)]
    res1 = np.zeros((m, n))
    res0 = np.zeros((m, n))

    used_budget = budget.copy()
    #order = np.random.permutation(m)
    order = range(m)
    for j in range(m):
        i = order[j]
        if real_match[i] == n:
            continue
        used_budget, temp = settle(real_match[i], used_budget, outcomes[i][real_match[i]], costs[i][real_match[i]])
        if real_match[i] == match1[i]:
            res1[i][match1[i]] += temp / p[i][match1[i]]
        if real_match[i] == match0[i]:
            res0[i][match0[i]] += temp / p[i][match0[i]]
    return res1, res0


def optimize(outcomes, budget, costs):
    x = cp.Variable(outcomes.shape)
    objective = cp.Minimize(cp.sum(cp.multiply(outcomes**2, cp.exp(-x))))
    constraints = [cp.sum(cp.exp(x), axis=1) <= 1, cp.sum(cp.multiply(costs, cp.exp(x)), axis=0) <= budget]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver='ECOS', max_iters=200)
    return np.array(np.exp(x.value))


def vars(outcomes, x):
    res = 0.0
    for i in range(m):
        for j in range(n):
            if x[i][j] > epsilon:
                res += outcomes[i][j]**2 * (1 / x[i][j] - 1)
        for j in range(n):
            for k in range(j + 1, n):
                res += 2 * outcomes[i][j] * outcomes[i][k]
    return res


def MSE(outcomes, match1, match0, x, p1, p2):
    res = 0.0

    for i in range(m):
        res += outcomes[i][match1[i]]**2 * (p1[i][match1[i]] /
                                            (x[i][match1[i]]**2) - 2 * p1[i][match1[i]] / x[i][match1[i]] + 1)
        res += outcomes[i][match0[i]]**2 * (p1[i][match0[i]] /
                                            (x[i][match0[i]]**2) - 2 * p1[i][match0[i]] / x[i][match0[i]] + 1)

    for i in range(m):
        for j in range(i + 1, m):

            res -= 2 * outcomes[i][match0[i]] * outcomes[j][match0[j]] * (p1[i][match0[i]] / x[i][match0[i]] +
                                                                          p1[j][match0[j]] / x[j][match0[j]] - 1 -
                                                                          (p1[j][match0[j]] * p1[i][match0[i]]) /
                                                                          (x[j][match0[j]] * x[i][match0[i]]))
            res -= 2 * outcomes[i][match1[i]] * outcomes[j][match1[j]] * (p1[i][match1[i]] / x[i][match1[i]] +
                                                                          p1[j][match1[j]] / x[j][match1[j]] - 1 -
                                                                          (p1[j][match1[j]] * p1[i][match1[i]]) /
                                                                          (x[j][match1[j]] * x[i][match1[i]]))
            '''
            res -= 2 * outcomes[i][match0[i]] * outcomes[j][match0[j]] * (
                p1[i][match0[i]] / x[i][match0[i]] + p1[j][match0[j]] / x[j][match0[j]] - 1 - p2[i + m][j + m] /
                (x[j][match0[j]] * x[i][match0[i]]))
            res -= 2 * outcomes[i][match1[i]] * outcomes[j][match1[j]] * (p1[i][match1[i]] / x[i][match1[i]] +
                                                                          p1[j][match1[j]] / x[j][match1[j]] - 1 - p2[i][j] /
                                                                          (x[j][match1[j]] * x[i][match1[i]]))
            '''
    for i in range(m):
        for j in range(m):
            res += 2 * outcomes[i][match0[i]] * outcomes[j][match1[j]] * (p1[i][match0[i]] / x[i][match0[i]] +
                                                                          p1[j][match1[j]] / x[j][match1[j]] - 1)
            if i != j:
                res -= 2 * outcomes[i][match0[i]] * outcomes[j][match1[j]] * (p1[j][match1[j]] * p1[i][match0[i]]) / (
                    x[j][match1[j]] * x[i][match0[i]])

    return res


def run_experiment(outcomes, costs, budget, match1, match0, prob, name):
    res = []
    count1 = np.zeros((m, n))
    count0 = np.zeros((m, n))
    count2 = np.zeros((2 * m, 2 * m))
    for i in range(N):
        temp1, temp0 = estimator2(outcomes, costs, budget, match1, match0, prob)
        res.append(np.sum(temp0 - temp1))
        '''
        temp1 = np.where(temp1 > epsilon, 1, 0)
        temp0 = np.where(temp0 > epsilon, 1, 0)
        count1 += temp1
        count0 += temp0
        
        for j in range(m):
            for k in range(m):
                count2[k][j] += temp1[k][match1[k]] * temp1[j][match1[j]]
                count2[k][j + m] += temp1[k][match1[k]] * temp0[j][match0[j]]
                count2[k + m][j] += temp0[k][match0[k]] * temp1[j][match1[j]]
                count2[k + m][j + m] += temp0[k][match0[k]] * temp0[j][match0[j]]
        '''
    #np.savetxt(name + " count.txt", (count1 + count0) / N)
    #np.savetxt(name + " p1_value.txt", (count1 + count0) / (opt1 + epsilon) / N)
    #np.savetxt(name + " p2_value.txt", count2 / N)
    return np.mean(res), np.std(res)
    #print("mean: ", np.mean(res0) - np.mean(res1))
    #print("var: ", np.var(np.array(res0) - np.array(res1)))
    #print("real MSE: ", np.mean(np.square(np.array(res0) - np.array(res1) - ground_truth)))
    #print("theory MSE: ", MSE(outcomes, match1, match0, prob, (count0 + count1) / N, count2 / N))
    print("----------------------------------------")


if __name__ == "__main__":
    res0 = []
    res1 = []
    res2 = []
    for t in range(100):
        print("episode " + str(t) + " start")
        print("--------------------------")
        m = 5 * (t + 1)
        match0 = np.random.randint(0, n, size=m)
        match1 = np.random.randint(0, n, size=m)
        for i in range(m):
            if match0[i] == match1[i]:
                match0[i] = (match1[i] + 1) % n
        costs = np.random.random((m, n))

        w0 = np.concatenate((2 * abs(np.random.normal(size=(m // 2, 1))), 4 * abs(np.random.normal(size=((m + 1) // 2, 1)))),
                            axis=0)
        w1 = np.concatenate((abs(np.random.normal(size=(m // 2, 1))), 2 * abs(np.random.normal(size=((m + 1) // 2, 1)))),
                            axis=0)
        outcomes = vec2mat(match1, w1) + vec2mat(match0, w0)
        budget = set_budget(match1, match0, costs)
        ground_truth = cal_outcome(w0, costs, budget, match0) - cal_outcome(w1, costs, budget, match1)
        res0.append(ground_truth)
        print(ground_truth)
        abtests = (vec2mat(match0, np.ones(len(match0))) + vec2mat(match1, np.ones(len(match1)))) / 2
        opt1 = optimize(outcomes, budget, costs)
        res1.append(run_experiment(outcomes, costs, budget, match1, match0, opt1, "offline" + str(t)))
        opt2 = np.zeros((m, n))
        for i in range(1, m + 1):
            temp = optimize(outcomes[:i], budget * i / m, costs[:i])
            opt2[i - 1] = temp[i - 1]
        '''
        opt3 = np.zeros((m, n))
        for i in range(1, m + 1):
            temp = optimize(outcomes[:i], budget * i / m * (1 - t * n / m), costs[:i])
            opt3[i - 1] = temp[i - 1]
        run_experiment(outcomes, costs, budget, match1, match0, opt3, "online2 " + str(t))
        '''
        res2.append(run_experiment(outcomes, costs, budget, match1, match0, opt2, "online" + str(t)))
        #res2.append(run_experiment(outcomes, costs, budget, match1, match0, opt1, "offline" + str(t)))
        np.savetxt("ground_truth_3.txt", res0)
        np.savetxt("offline_3.txt", res1)
        np.savetxt("online_3.txt", res2)