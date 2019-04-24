from random import shuffle
import math
import numpy as np

def MAP(S, T, variance, gamma, d):
    w = [0.0 for _ in range(len(S[0]) - 1)]
    m = float(len(S))

    for t in range(T):
        shuffle(S)
        for s in S:
            gradient = []
            for i in range(len(w)):
                dp = np.dot(w, s[:-1])
                e = math.exp(dp * -s[-1])
                gradient.append(-(s[-1] * s[i] * m * e)/(1 + e) + (s[i]/(2 * variance)))

            w = w + s[-1] * gamma * np.array(gradient)

        gamma = gamma / (1 + (gamma/d) * t)

    return w

def MLE(S, T, variance, gamma, d):
    w = [0.0 for _ in range(len(S[0]) - 1)]
    m = float(len(S))

    for t in range(T):
        shuffle(S)
        for s in S:
            gradient = []
            for i in range(len(w)):
                dp = np.dot(w, s[:-1])
                e = math.exp(dp * -s[-1])
                gradient.append(-(s[-1] * s[i] * m * e)/(1 + e))

            w = w + s[-1] * gamma * np.array(gradient)

        gamma = gamma / (1 + (gamma/d) * t)

    return w

def LR_Test(S, w):
    wrong = 0
    for s in S:
        guess = np.dot(s[:-1], w)
        
        if guess > 0 and s[-1] == 1:
            pass
        elif guess < 0 and s[-1] == -1:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))
