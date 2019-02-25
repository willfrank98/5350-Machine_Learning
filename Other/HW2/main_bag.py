from ada import *
from bagging import *
from forest import *
import random

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
dataset = "bank"
# function = "InfoGain"
#maxdepth = int(sys.argv[3])

attrFile = open(dataset + "/data-desc.txt")
attrFile.readline()
attrFile.readline()
# reads the line, uses split/join to remove all whitespace, splits on commas
labels = "".join(attrFile.readline().split()).split(',')

attrFile.readline()
attrFile.readline()
attrFile.readline()

Attributes = {}
attrList = []

line = attrFile.readline()
while line != "\n":
    splitLine = line.split(':')
    attr = splitLine[0]
    attrList.append(attr)
    attrVals = "".join(splitLine[1].split()).split(',')
    Attributes[attr] = attrVals
    line = attrFile.readline()

attrList.append("Label")

# gets a list of all attributes that are numerical rather than categorical
numericList = [A for A in attrList if A in Attributes and Attributes[A][0] == "(numeric)"]
# unknownList = [A for A in attrList if A in Attributes and Attributes[A][0] == "unknown"]

# opens training data
S_train = []
numericalLists = {}
# unknownsLists = {}
with open(dataset + "/train.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            # if this is a numerical attribute, add it to the list that calculates medians
            attrName = attrList[i]
            if attrName in numericList:
                if attrName not in numericalLists:
                    numericalLists[attrName] = []
                numericalLists[attrName].append(float(attr))
            # if this attribute can be unknown then track the most common attribute value
            # if attrName in unknownList and attr != "unknown":
            #     if attrName not in unknownsLists:
            #         unknownsLists[attrName] = []
            #     unknownsLists[attrName].append(attr)
            example[attrList[i]] = attr
            i += 1
        S_train.append(example)

medianList = {}
for name, arr in numericalLists.items():
    medianList[name] = calc_median(arr)

# unknownReplace = {}
# for name, arr in unknownsLists.items():
#     unknownReplace[name] = max(set(arr), key=arr.count)

# convert numerical data into binary
# convert unknowns into most common value
# assigns uniform distribution weight
for s in S_train:
    for attr in numericList:
        if s[attr] >= numericalLists[attr]:
            s[attr] = "1"
        elif s[attr] < numericalLists[attr]:
            s[attr] = "-1"
    # for attr in unknownList:
    #     if s[attr] == "unknown":
    #         s[attr] = unknownReplace[attr]
    s["Weight"] = 1/float(len(S_train))

S_test = []
with open(dataset + "/test.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            name = attrList[i]
            # convert numerical data into binary
            if name in numericList:
                val = float(attr)
                if val >= numericalLists[name]:
                    attr = "1"
                elif val < numericalLists[name]:
                    attr = "-1"
            # convert unknown into most common training value
            # if name in unknownList and attr == "unknown":
            #     attr = unknownReplace[name]
            example[name] = attr
            i += 1
        S_test.append(example)

# fix (numeric) Attributes
for attr in numericList:
    Attributes[attr] = ["-1", "1"]



asodufh = Random_Forest_Train(S_train, Attributes, 10, 2)



predictors = []
for _ in range(0, 100):
    copy_S = list(S_train)
    new_S = []
    for i in range(0, 1000):
        rand = random.randint(0, len(copy_S) - 1)
        new_S.append(copy_S[rand])
        del copy_S[rand]
    predictor = Bagging_Train(new_S, Attributes, 1000)
    predictors.append(predictor)

pass




# for T in range(1, 1001, 10):
#     hypothesis = Bagging(S_train, Attributes, T)
#     err_train = Bagging_Test(hypothesis, S_train)
#     err_test = Bagging_Test(hypothesis, S_test)
#     print "T = " + str(T-1) + ": " + str(err_train) + ",\t" + str(err_test)
#     reset_weights(S_train)
