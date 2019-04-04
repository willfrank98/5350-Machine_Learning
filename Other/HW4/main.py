from svm import SVM_Primal_Test, SVM_Primal_Train, SVM_Dual_Test, SVM_Dual_Train

bank_note = "bank-note/"

f = open(bank_note + "train.csv")

Attributes = ["Variance", "Skewness", "Curtosis", "Entropy", "Label"]

S_train = []
for line in f:
    attrs = line.strip().split(',')
    example = {}
    for i in range(0, len(Attributes)):
        if Attributes[i] == "Label":
            attrs[i] = 1 if attrs[i] == '1' else -1
        example[Attributes[i]] = float(attrs[i])
    S_train.append(example)

f = open(bank_note + "test.csv")
S_test = []
for line in f:
    attrs = line.strip().split(',')
    example = {}
    for i in range(0, len(Attributes)):
        if Attributes[i] == "Label":
            attrs[i] = 1 if attrs[i] == '1' else -1
        example[Attributes[i]] = float(attrs[i])
    S_test.append(example)

Attributes.remove("Label")

test = []
train = []

#temp_s = S_train[:20]

C = [100, 500, 700]
for c in C:
    w, b = SVM_Dual_Train(S_train, Attributes, C = float(c)/873) 
    #err_train = SVM_Dual_Test(w, b, S_train, Attributes)
    #err_test = SVM_Dual_Test(w, b, S_test, Attributes)
    print "w: " + str(w)
    print "b: " + str(b)
    # print "w: " + str([round(wi, 3) for wi in w])
    # train.append(round(err_train, 3))
    # test.append(err_test)

# print str(train).replace(', ', ' & ')
# print str(test).replace(', ', ' & ')
