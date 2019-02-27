from lms import Batch_LMS, Stochastic_LMS, get_cost 

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

test_cost = get_cost(S_test, Attributes, w)
print test_cost

w = Stochastic_LMS(S_train, Attributes, .1, .000001)

test_cost = get_cost(S_test, Attributes, w)
print test_cost
