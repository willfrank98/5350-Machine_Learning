w = [0, 0, 0, 0]
# b = 0

S = [[1, -1, 2, 1, 1],
     [1, 1, 3, 1, 4],
     [-1, 1, 0, 1, -1],
     [1, 2, -4, 1, -2], 
     [3, -1, -1, 1, 0]]

R = .1

while 0 < 1:
    storedW = list(w)
    for i in range(0, len(S)):
        newW = []
        gradient = []
        for j in range(0, 4):
            vals = S[i][:4]
            prediction = 0.0
            for k in range(0, len(w)):
                prediction += w[k] * vals[k]
            label = S[i][4]
            error = label - prediction
            xij = S[i][j]
            adjustment = error * xij
            gradient.append(-adjustment)
            adjustment *= R
            newW.append(w[j] + adjustment)
        print str(gradient)
        w = newW


    norm = -1
    for i in range(0, len(w)):
        diff = abs(storedW[i] - w[i])
        if diff > norm:
            norm = diff
