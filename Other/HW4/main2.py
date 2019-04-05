from svm import SVM_Dual_Test, SVM_Dual_Train
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

print "\nGenerating output for 3a"
C = [100, 500, 700]
for c in C:
    w, b = SVM_Dual_Train(S_train, C = float(c)/873) 
    print "w = " + str(w)
    print "b = " + str(b)