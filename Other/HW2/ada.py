from id3 import id3_with_weight, get_label, id3_weighted_err
import math

def AdaBoost(S, Attributes, T):
    trees = []
    alphas = []
    for _ in range(0, T):
        tree = id3_with_weight(S, Attributes, 1, 0)
        #render(tree, None)
        trees.append(tree)

        # normalize values
        norm = 0.0        
        epsilon = id3_weighted_err(tree, S)
        alpha = .5 * math.log((1 - epsilon)/epsilon)
        alphas.append(alpha)
        for s in S:
            label = get_label(s, tree)
            if label != s["Label"]:
                newWeight = s["Weight"] * math.exp(alpha)
                s["Weight"] = newWeight
            else:
                newWeight = s["Weight"] * math.exp(-alpha)
                s["Weight"] = newWeight
            norm += newWeight

        for s in S:
            s["Weight"] /= norm

    return (trees, alphas)


def AdaBoost_Test(Hypothesis, S):
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
