from id3 import id3_with_weight, get_label, id3_weighted_err
import math

def AdaBoost(S, Attributes, T):
    trees = []
    alphas = []
    for _ in range(0, T):
        tree = id3_with_weight(S, Attributes, 1, 0)
        trees.append(tree)

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

        # normalize values
        for s in S:
            s["Weight"] /= norm

    return (trees, alphas)

def AdaBoost_print(S_train, S_test, Attributes, T):
    trees = []
    alphas = []
    f = open("adaboost_out2.txt", "w")
    for _ in range(0, T):
        tree = id3_with_weight(S_train, Attributes, 1, 0)
        trees.append(tree)

        norm = 0.0        
        epsilon = id3_weighted_err(tree, S_train)
        f.write(str(_) + "\t" + str(epsilon) + "\t" + str(id3_weighted_err(tree, S_test)) + "\n")
        alpha = .5 * math.log((1 - epsilon)/epsilon)
        alphas.append(alpha)
        for s in S_train:
            label = get_label(s, tree)
            if label != s["Label"]:
                newWeight = s["Weight"] * math.exp(alpha)
                s["Weight"] = newWeight
            else:
                newWeight = s["Weight"] * math.exp(-alpha)
                s["Weight"] = newWeight
            norm += newWeight

        # normalize values
        for s in S_train:
            s["Weight"] /= norm

    return (trees, alphas)

def AdaBoost_Test(Hypothesis, S):
    wrong = 0
    for s in S:
        prediction = 0
        for tree, weight in zip(Hypothesis[0], Hypothesis[1]):
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

def avg_error(Hypothesis, S):
    total = 0.0
    for tree in Hypothesis[0]:
        temp = 0
        for s in S:
            label = get_label(s, tree)
            if s["Label"] != label:
                temp += 1

        total += temp/float(len(S))

    return total/float(len(Hypothesis[0]))