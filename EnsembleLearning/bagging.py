from id3 import id3_with_weight, id3_weighted_err, get_label
import random

def Bagging_Train(S, Attributes, T):
    M = len(S)/2
    predictions = []
    weights = []
    for _ in range(0, T):
        new_S = [random.choice(S) for __ in range(0, M)]

        tree = id3_with_weight(new_S, Attributes, None, 0)
        predictions.append(tree)

    return predictions


def Bagging_Test(Hypothesis, S):
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