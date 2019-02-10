from id3 import ID3_Train, ID3_Test
from render import render


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
    attrVals = "".join(splitLine[1].replace('.', '').split()).split(',')
    Attributes[attr] = attrVals
    line = attrFile.readline()

attrList.append("Label")

S_train = []
with open(dataset + "/train.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            example[attrList[i]] = attr
            i += 1
        S_train.append(example)

S_test = []
with open(dataset + "/test.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            example[attrList[i]] = attr
            i += 1
        S_test.append(example)

print "DataSet:", dataset, "Algorithm:", function
for i in range(1, 7):
    Tree = ID3_Train(S_train, Attributes, function, i)
    print "Depth:", i, "Training error:", ID3_Test(Tree, S_train), "Testing error:", ID3_Test(Tree, S_test)
