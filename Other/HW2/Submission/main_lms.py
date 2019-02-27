from lms import Batch_LMS, Stochastic_LMS, get_cost
#import numpy as np

### MAIN ###
dataset = "concrete"

Attributes = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "Label"]

# opens training data
S_train = []
with open(dataset + "/train.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            example[Attributes[i]] = float(attr)
            i += 1
        S_train.append(example)

S_test = []
with open(dataset + "/test.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            example[Attributes[i]] = float(attr)
            i += 1
        S_test.append(example)

Attributes.remove("Label")

## Calculates optimal weights ##
# x = []
# y = []

# for i in range(0, 7):
#     tempX = []
#     for s in S_train:
#         tempX.append(s[Attributes[i]])
#     x.append(tempX)

# for s in S_train:
#     y.append([s["Label"]])

# X = np.array(x)
# Y = np.array(y)

# Xt = np.transpose(X)

# X2 = np.matmul(X, Xt)

# Xi = np.linalg.inv(X2)

# XY = np.matmul(X, Y)

# final = np.matmul(Xi, XY)

# print str(final)

print "Generating data for problem 4a"

w = Batch_LMS(S_train, Attributes, .0145, 0.000001)

test_cost = get_cost(S_test, Attributes, w)
print test_cost


print "Generating data for problem 4b"

w = Stochastic_LMS(S_train, Attributes, .1, .000001)

test_cost = get_cost(S_test, Attributes, w)
print test_cost
