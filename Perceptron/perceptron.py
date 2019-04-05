import random

def Perceptron_Standard(S, Attributes, R, MaxEpochs):
    w = [0 for _ in range(0, len(Attributes))]
    for _ in range(0, MaxEpochs):
        random.shuffle(S)

        for s in S:
            guess = 0.0
            for i in range(0, len(w)):
                guess += float(s[Attributes[i]]) * w[i]
            guess *= s["Label"]
            if guess <= 0:
                for i in range(0, len(w)):
                    w[i] += float(s[Attributes[i]]) * s["Label"] * R
            
    return w

def Perceptron_Voted(S, Attributes, R, MaxEpochs):
    w = [0 for _ in range(0, len(Attributes))]
    C = []
    W = []
    c = 1
    for _ in range(0, MaxEpochs):
        random.shuffle(S)

        for s in S:
            guess = 0.0
            for i in range(0, len(w)):
                guess += float(s[Attributes[i]]) * w[i]
            guess *= s["Label"]
            if guess <= 0:
                if w[0] != 0 and w[-1] != 0: # makes sure the very first 0 weight vector is not appended 
                    W.append(list(w))
                    C.append(c)
                c = 1
                for i in range(0, len(w)):
                    w[i] += float(s[Attributes[i]]) * s["Label"] * R
            else:
                c += 1
            
    return W, C

def Perceptron_Average(S, Attributes, R, MaxEpochs):
    w = [0 for _ in range(0, len(Attributes))]
    a = list(w)
    for _ in range(0, MaxEpochs):
        random.shuffle(S)

        for s in S:
            guess = 0.0
            for i in range(0, len(w)):
                guess += float(s[Attributes[i]]) * w[i]
            guess *= s["Label"]
            if guess <= 0:
                for i in range(0, len(w)):
                    w[i] += float(s[Attributes[i]]) * s["Label"] * R
            for i in range(0, len(w)):
                    a[i] += w[i]
            
    return a

def Perceptron_Test(W, S, Attributes):
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

def Perceptron_Test_Voted(WC, S, Attributes):
    wrong = 0
    for s in S:
        guess = 0.0
        for i in range(0, len(WC[0])):
            w = WC[0][i]
            c = WC[1][i]
            tmp_guess = 0.0
            for j in range(0, len(w)):
                tmp_guess += w[j] * float(s[Attributes[j]])
            tmp_guess = 1 if tmp_guess > 0 else -1
            guess += tmp_guess * c                
        
        if guess > 0 and s["Label"] == 1:
            pass
        elif guess < 0 and s["Label"] == -1:
            pass
        else:
            wrong += 1

    return wrong/float(len(S))