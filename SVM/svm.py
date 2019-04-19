from random import shuffle
from scipy.optimize import minimize
import numpy as np
import math

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

def SVM_Primal_Train_A(S, Attributes, C, num_iter, gamma, d): 
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

def SVM_Primal_Train_B(S, Attributes, C, num_iter, gamma, d): 
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

        gamma = gamma / (1 + t)
    return w


def SVM_Dual_Train(S, C):
    def main_func(x):
        ret_val = 0
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                ret_val += x[i] * x[j] * np.dot(S[i], S[j])
        return (ret_val / 2) - sum(x)

    def cons_func(x):
        ret_val = 0
        for i in xrange(len(x)):
            ret_val += x[i] * S[i][-1]
        return ret_val

    bounds = [(0, C) for _ in range(len(S))]
    w = [0 for _ in range(len(S))]
    constraints = ({'type':'eq', 'fun':cons_func})

    result = minimize(main_func, w, method='SLSQP', bounds=bounds, constraints=constraints)
    A = result.x

    w = [0 for _ in range(len(S[0]) - 1)]
    for i in range(len(S)):
        for j in range(len(S[0]) - 1):
            w[j] += A[i] * S[i][-1] * S[i][j]

    count = 0
    for i in range(len(A)):
        if A[i] > 0 and A[i] < C:
            b = 0
            for j in range(len(w)):
                b += w[j] * S[i][j]
            count += 1

    return w, b/count


def kernel(x1, x2, gamma):
        norm = np.linalg.norm(x1-x2)
        norm = -(norm**2)
        norm /= gamma
        norm = math.exp(norm)
        return norm

def SVM_Kernel_Train(S, C, gamma):
    def main_func(x):
        ret_val = 0
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                ret_val += x[i] * x[j] * S[i][-1] * S[j][-1] * kernel(S[i][:-1], S[j][:-1], gamma)
        return (ret_val / 2) - sum(x)

    def cons_func(x):
        ret_val = 0
        for i in xrange(len(x)):
            ret_val += x[i] * S[i][-1]
        return ret_val

    bounds = [(0, C) for _ in range(len(S))]
    w = [0 for _ in range(len(S))]
    constraints = ({'type':'eq', 'fun':cons_func})

    result = minimize(main_func, w, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def SVM_Kernel_Test(a, S, S_train, gamma):
    wrong = 0
    for s in S:
        guess = 0
        for i in range(len(a)):
            guess += a[i] * S_train[i][-1] * kernel(S_train[i][:-1], s[:-1], gamma)
        
        if guess > 0 and s[-1] == 1:
            pass
        elif guess < 0 and s[-1] == -1:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))

def SVM_Primal_Test(W, S, Attributes):
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

def SVM_Dual_Test(w, b, S, Attributes):
    wrong = 0
    for s in S:
        guess = b
        for i in range(len(w)):
            guess += float(s[Attributes[i]]) * w[i]
        
        if guess > 0 and s["Label"] == 1:
            pass
        elif guess < 0 and s["Label"] == -1:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))

def SVM_Count(a):
    return sum([1 for alpha in a if alpha > 0])