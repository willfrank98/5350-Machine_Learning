import random
from bagging import Bagging_Train, Bagging_Test
from forest import Random_Forest_Train, Random_Forest_Test
from ada import AdaBoost, AdaBoost_Test
from id3 import id3_with_weight, id3_weighted_err

def calc_median(arr):
    n = len(arr)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(arr)[n//2]
    else:
            return sum(sorted(arr)[n//2-1:n//2+1])/2.0

def reset_weights(S):
    weight = 1/float(len(S))
    for s in S:
        s["Weight"] = weight


### MAIN ###
attrList = ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","Label"]

Attributes = {"LIMIT_BAL":["(numeric)"],
              "SEX":["1", "2"],
              "EDUCATION":["0", "1", "2", "3", "4", "5", "6"],
              "MARRIAGE":["0", "1", "2", "3"],
              "AGE":["(numeric)"],
              "PAY_0":["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
              "PAY_2":["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
              "PAY_3":["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
              "PAY_4":["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
              "PAY_5":["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
              "PAY_6":["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
              "BILL_AMT1":["(numeric)"],
              "BILL_AMT2":["(numeric)"],
              "BILL_AMT3":["(numeric)"],
              "BILL_AMT4":["(numeric)"],
              "BILL_AMT5":["(numeric)"],
              "BILL_AMT6":["(numeric)"],
              "PAY_AMT1":["(numeric)"],
              "PAY_AMT2":["(numeric)"],
              "PAY_AMT3":["(numeric)"],
              "PAY_AMT4":["(numeric)"],
              "PAY_AMT5":["(numeric)"],
              "PAY_AMT6":["(numeric)"]}

# gets a list of all attributes that are numerical rather than categorical
numericList = [A for A in attrList if A in Attributes and Attributes[A][0] == "(numeric)"]

# opens training data
S_train = []
numericalLists = {}
with open("default of credit card clients.csv") as f:
    f.readline()
    f.readline()
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            if i == 0:
                i += 1
                continue
            # if this is a numerical attribute, add it to the list that calculates medians
            attrName = attrList[i]
            if attrName in numericList:
                if attrName not in numericalLists:
                    numericalLists[attrName] = []
                numericalLists[attrName].append(float(attr))
            if attrList[i] == "Label":
                attr = "yes" if attr == "1" else "no"
            example[attrList[i]] = attr
            i += 1
        S_train.append(example)

medianList = {}
for name, arr in numericalLists.items():
    medianList[name] = calc_median(arr)

# convert numerical data into binary
# assigns uniform distribution weight
for s in S_train:
    for attr in numericList:
        if s[attr] >= numericalLists[attr]:
            s[attr] = "1"
        elif s[attr] < numericalLists[attr]:
            s[attr] = "-1"
    s["Weight"] = 0

# fix (numeric) Attributes
for attr in numericList:
    Attributes[attr] = ["-1", "1"]

#randomly choose 6000 examples for test data
S_test = []
for i in range(0, 6000):
    n = random.randint(0, len(S_train)-1)
    S_test.append(S_train[n])
    del S_train[n]

reset_weights(S_train)
reset_weights(S_test)

# Bagged trees
f = open("bonus_out.txt", "w")
f.write("Iter\tTrain\tTest\n")
for T in range(1, 1050, 50):
    hypothesis = Bagging_Train(S_train, Attributes, T)
    err_train = Bagging_Test(hypothesis, S_train)
    err_test = Bagging_Test(hypothesis, S_test)
    print "T = " + str(T-1) + ": " + str(err_train) + ", " + str(err_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")

f.write("\n\n")

# random forest 
f.write("Iter\tTrain\tTest\n")
for T in range(1, 1050, 50):
    hypothesis = Random_Forest_Train(S_train, Attributes, T, 4)
    err_train = Random_Forest_Test(hypothesis, S_train)
    err_test = Random_Forest_Test(hypothesis, S_test)
    print "T = " + str(T-1) + ": " + str(err_train) + ", " + str(err_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")

f.write("\n\n")

# adaboost
f.write("Iter\tTrain\tTest\n")
for T in range(1, 1050, 50):
    hypothesis = AdaBoost(S_train, Attributes, T)
    err_train = AdaBoost_Test(hypothesis, S_train)
    err_test = AdaBoost_Test(hypothesis, S_test)
    print str(T-1) + "\t" + str(err_train) + "\t" + str(err_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

f.write("\n\n")

# fully expanded tree
reset_weights(S_train)
reset_weights(S_test)
tree = id3_with_weight(S_train, Attributes, None, 0)
err_train = id3_weighted_err(tree, S_train)
err_test = id3_weighted_err(tree, S_test)
print "T = " + str(T-1) + "\t" + str(err_train) + "\t" + str(err_test)
f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")