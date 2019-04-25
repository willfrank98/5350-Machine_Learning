from logres import MAP, MLE, LR_Test
from nn import backprop
import numpy as np

bank_note = "bank-note/"

f = open(bank_note + "train.csv")
S_train = []
for line in f:
    attrs = line.strip().split(',')
    example = [float(s) for s in attrs[:-1]]
    if attrs[-1] == '1':
        example.append(1)
    else:
        example.append(-1)
    S_train.append(example)

f = open(bank_note + "test.csv")
S_test = []
for line in f:
    attrs = line.strip().split(',')
    example = [float(s) for s in attrs[:-1]]
    if attrs[-1] == '1':
        example.append(1)
    else:
        example.append(-1)
    S_test.append(example)

S_test = np.array(S_test)
# S_train = np.array(S_train)

print "Output for 2a"

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
for v in variances:
    temp = np.array(S_train)
    w = MAP(temp, 100, v, 0.00001, 0.0001)
    err_train = LR_Test(S_train, w)
    err_test = LR_Test(S_test, w)
    print str(v) + " & " + str(err_train) + " & " + str(err_test) + " \\\\"


print "Output for 2b"

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
for v in variances:
    temp = np.array(S_train)
    w = MLE(temp, 100, v, 0.00001, 0.0001)
    err_train = LR_Test(S_train, w)
    err_test = LR_Test(S_test, w)
    print str(v) + " & " + str(err_train) + " & " + str(err_test) + " \\\\"