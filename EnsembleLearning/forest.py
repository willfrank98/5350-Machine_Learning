import random
from id3 import *

def Random_Forest_Train(S, Attributes, T, num_features):
    M = len(S)/2
    predictions = []
    for _ in range(0, T):
        new_S = [random.choice(S) for __ in range(0, M)]

        tree = id3_rand_learn(new_S, Attributes, num_features)
        predictions.append(tree)

    return predictions

def Random_Forest_Test(Hypothesis, S):
    wrong = 0
    for s in S:
        prediction = 0
        for tree in Hypothesis:
            label = get_label(s, tree)
            label = 1 if label == "yes" else -1
            prediction += label

        if s["Label"] == "yes" and prediction > 0:
            pass
        elif s["Label"] == "no" and prediction < 0:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))