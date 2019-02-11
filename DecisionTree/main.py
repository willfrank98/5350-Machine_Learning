from id3 import ID3_Train, ID3_Test
from render import render

def open_dataset(FileName, NumericList, AttributesList):
    
    return S

def calc_median(arr):
    n = len(arr)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(arr)[n//2]
    else:
            return sum(sorted(arr)[n//2-1:n//2+1])/2.0


### MAIN ###
dataset = "bank"
function = "InfoGain"
maxDepth = 1
output = "cars"

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

# opens training data
S_train = []
numericalLists = {}
with open(dataset + "/train.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            # if this is a numerical attribute, add it to the list that calculates medians
            if attrList[i] in numericList:
                attrName = attrList[i]
                if attrName not in numericalLists:
                    numericalLists[attrName] = []
                numericalLists[attrName].append(float(attr))
            example[attrList[i]] = attr
            i += 1
        S_train.append(example)

medianList = {}
for name, arr in numericalLists.items():
    medianList[name] = calc_median(arr)

# convert numerical data into binary
for s in S_train:
    for attr in numericList:
        if s[attr] >= numericalLists[attr]:
            s[attr] = "1"
        elif s[attr] < numericalLists[attr]:
            s[attr] = "-1"

S_test = []
with open(dataset + "/test.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            if attrList[i] in numericList:
                val = float(attr)
                # convert numerical data into binary
                if val >= numericalLists[attr]:
                    attr = "1"
                elif s[attr] < numericalLists[attr]:
                    attr = "-1"
            example[attrList[i]] = attr
            i += 1
        S_test.append(example)

print "DataSet:", dataset, "Algorithm:", function
for i in range(1, 7):
    Tree = ID3_Train(S_train, Attributes, function, i)
    render(Tree, None)
    print "Depth:", i, "Training error:", ID3_Test(Tree, S_train), "Testing error:", ID3_Test(Tree, S_test)