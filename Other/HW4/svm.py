from random import shuffle
from scipy.optimize import minimize

def SVM_Primal_Train(S, Attributes, C, num_iter, gamma, d): 
    w = [0 for _ in range(len(Attributes))]

    for t in range(0, num_iter):
        shuffle(S)
        for s in S:
            prod = 0
            for i in range(len(w)):
                prod += s[Attributes[i]] * w[i]
            prod *= s['Label']

            if prod <= 1:
                w = [(1 - gamma) * w[i] + (gamma * C * len(S) * s['Label'] * s[Attributes[i]]) for i in range(len(w))]
            else:
                w = [(1 - gamma) * w[i] for i in range(len(w))]

        gamma = gamma / (1 + gamma/d*t)
    return w


def SVM_Dual_Train(S, Attributes, C):
    def main_func(x):
        ret_val = 0
        for i in range(len(x)):
            for j in range(len(x)):
                ret_val += S[i]['Label'] * S[j]['Label'] * x[i] * x[j] * dot_prod(S[i], S[j], Attributes)
        return (ret_val / 2) - sum(x)

    def cons_func(x):
        ret_val = 0
        for i in range(len(x)):
            ret_val += x[i] * S[i]['Label']
        return ret_val

    bounds = [(0, C) for _ in range(len(S))]
    w = [0 for _ in range(len(S))]
    constraints = ({'type':'eq', 'fun':cons_func})

    result = minimize(main_func, w, method='L-BFGS-B', bounds=bounds, constraints=constraints)
    A = result.x

    w = [0 for _ in range(len(Attributes))]
    for i in range(len(S)):
        for j in range(len(Attributes)):
            w[j] += A[i] * S[i]['Label'] * S[i][Attributes[j]]

    return w

def dot_prod(x1, x2, Attributes):
    tol = 0
    for i in range(len(Attributes)):
        tol += x1[Attributes[i]] * x2[Attributes[i]]
    return tol

def SVM_Test(W, S, Attributes):
    wrong = 0
    for s in S:
        guess = 0.0
        for i in range(0, len(W)):
            guess += float(s[Attributes[i]]) * W[i]
        
        if guess > 0 and s["Label"] == 1:
            pass
        elif guess < 0 and s["Label"] == -1:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))