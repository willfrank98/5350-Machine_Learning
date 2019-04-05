from svm import SVM_Kernel_Test, SVM_Kernel_Train, SVM_Count
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
S_train = np.array(S_train)
newS_train = np.array(S_train[0:100])

print "\nGenerating output for 3b"
C = [100, 500, 700]
gamma = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]
for c in C:
    for g in gamma:
        a = SVM_Kernel_Train(newS_train, C = float(c)/873, gamma=g) 
        print "g = " + str(g) + ", c = " + str(c)
        test = SVM_Kernel_Test(a, S_test, newS_train, g)
        train = SVM_Kernel_Test(a, newS_train, newS_train, g)
        print "train = " + str(train)
        print "test = " + str(test)
        print "SV# = " + str(SVM_Count(a)/100.0)