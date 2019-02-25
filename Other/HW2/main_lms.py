from lms import Batch_LMS, Stochastic_LMS

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

w = Batch_LMS(S_train, Attributes, .0145, 0.000001)

w = Stochastic_LMS(S_train, Attributes, .2, .000001)
