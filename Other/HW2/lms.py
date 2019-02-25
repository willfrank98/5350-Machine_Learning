import random

def Batch_LMS(S, Attributes, R, Convergence):
    w = [0 for _ in range(0, len(Attributes))]
    f = open("batch_out.txt", "w")
    norm = 1000
    T = 0
    f.write("Iteration\tCost\n")
    print "Iteration\tCost"
    while norm > Convergence:
        cost = 0.0
        J_grad = []
        for j in range(0, len(Attributes)):
            temp = 0.0
            for s in S:
                vals = [s[attr] for attr in Attributes]
                prediction = 0.0
                for k in range(0, len(w)):
                    prediction += w[k] * vals[k]
                label = s["Label"]
                error = label - prediction
                xij = s[Attributes[j]]
                temp += error * xij
            J_grad.append(-temp)

        cost = get_cost(S, Attributes, w)
        f.write(str(T) + "\t" + str(cost/2) + "\n")
        print str(T) + "\t" + str(cost/2)
        T += 1

        newW = []
        for i in range(0, len(w)):
            newW.append(w[i] - (R * J_grad[i]))

        norm = -1
        for i in range(0, len(w)):
            gap = abs(w[i] - newW[i])
            if gap > norm:
                norm = gap

        w = newW

    f.write("Final W: " + str(w))
    print "Final W: " + str(w)
    return w 
    

def Stochastic_LMS(S, Attributes, R, Convergence):
    w = [0 for _ in range(0, len(Attributes))]
    f = open("stoch_out.txt", "w")
    f.write("Iteration\tCost\n")
    print "Iteration\tCost"
    norm = 1
    T = 0
    while norm > Convergence:
        #random.shuffle(S)
        storedW = list(w)
        for i in range(0, len(S)):
            newW = []
            for j in range(0, len(Attributes)):
                vals = [S[i][attr] for attr in Attributes]
                prediction = 0.0
                for k in range(0, len(w)):
                    prediction += w[k] * vals[k]
                label = S[i]["Label"]
                error = label - prediction
                xij = S[i][Attributes[j]]
                adjustment = error * xij * R
                newW.append(w[j] + adjustment)
            w = newW

        cost = get_cost(S, Attributes, w)
        f.write(str(T) + "\t" + str(cost/2) + "\n")
        print str(T) + "\t" + str(cost/2)
        T += 1

        norm = -1
        for i in range(0, len(w)):
            diff = abs(storedW[i] - w[i])
            if diff > norm:
                norm = diff

    f.write("Final W: " + str(w))
    print "Final W: " + str(w)
    return w


def get_cost(S, Attributes, w):
    cost = 0.0
    for s in S:
        vals = [s[attr] for attr in Attributes]
        prediction = 0.0
        for k in range(0, len(w)):
            prediction += w[k] * vals[k]
        label = s["Label"]
        error = label - prediction
        cost += pow(error, 2)

    return cost/2