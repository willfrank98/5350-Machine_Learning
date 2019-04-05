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
S_train = np.array(S_train[:200])

print "\nGenerating output for 3c"
gamma = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]
old_a = []
for g in gamma:
    a = SVM_Kernel_Train(S_train, C=500.0/873, gamma=g) 
    print "g = " + str(g)
    count = 0
    for i in range(len(old_a)):
        if a[i] > 0 and old_a[i] > 0:
            count += 1
    print "count = " + str(float(count)/len(a))
    old_a = a