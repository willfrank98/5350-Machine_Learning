from id3 import id3_with_weight, id3_weighted_err, get_label, render
import random

def Bagging(S, Attributes, T):
    M = len(S)/2
    predictions = []
    weights = []
    for _ in range(0, T):
        new_S = []
        for __ in range(0, M):
            rand = random.randint(0, len(S) - 1)
            new_S.append(S[rand])

        tree = id3_with_weight(new_S, Attributes, None, 0)
        #render(tree, "tree")
        predictions.append(tree)
        #weights.append(1 - id3_weighted_err(tree, S))

    return predictions


def Bagging_Test(Hypothesis, S):
    wrong = 0
    for s in S:
        prediction = 0
        for tree in Hypothesis:
            label = get_label(s, tree)
            label = 1 if label == "yes" else -1
            prediction += label

        prediction /= float(len(Hypothesis))
        if s["Label"] == "yes" and prediction > 0:
            pass
        elif s["Label"] == "no" and prediction < 0:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))