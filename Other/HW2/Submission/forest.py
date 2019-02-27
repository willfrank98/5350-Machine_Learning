import random
from id3 import *

def Random_Forest_Train(S, Attributes, T, num_features):
    M = len(S)/2
    predictions = []
    weights = []
    for _ in range(0, T):
        new_S = [random.choice(S) for __ in range(0, M)]

        tree = id3_rand_learn(new_S, Attributes, num_features)
        predictions.append(tree)
        weights.append(1)

    return predictions, weights

def Random_Forest_Test(Hypothesis, S):
    wrong = 0
    for s in S:
        prediction = 0
        for tree , weight in zip(Hypothesis[0], Hypothesis[1]):
            label = get_label(s, tree)
            label = 1 if label == "yes" else -1
            prediction += label * weight

        if s["Label"] == "yes" and prediction > 0:
            pass
        elif s["Label"] == "no" and prediction < 0:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))