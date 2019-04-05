from svm import SVM_Kernel_Test, SVM_Kernel_Train
import numpy as np

bank_note = "bank-note/"

f = open(bank_note + "train.csv")
S_train = []
for line in f:
    attrs = line.strip().split(',')
    example = [float(s) for s in attrs]
    S_train.append(example)

f = open(bank_note + "test.csv")
S_test = []
for line in f:
    attrs = line.strip().split(',')
    example = [float(s) for s in attrs]
    S_test.append(example)


S_test = np.array(S_test)
S_train = np.array(S_train[0:50])

C = [100, 500, 700]
gamma = [0.1, 0.5, 1, 2, 5, 10, 100]
for c in C:
    for g in gamma:
        w, b = SVM_Kernel_Train(S_train, C = float(c)/873, gamma=g) 
        print "g = " + str(g) + ", c = " + str(c)
        test = SVM_Kernel_Test(w, b, S_test)
        train = SVM_Kernel_Test(w, b, S_train)
        print "train = " + str(train)
        print "test = " + str(test)