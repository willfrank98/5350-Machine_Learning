import math


class Node:
    children = list()
    #parent = None
    label = ""
    splitsOn = ""
    prediction = ""


def ID3(S, Attributes, Label):
    #check if all labels are the same
    labelCheck = S[0]["Label"]
    allSame = True
    for s in S:
        if s["Label"] != labelCheck:
            allSame = False
            break

    if allSame:
        # return a leaf node with the label
        leaf = Node()
        leaf.prediction = labelCheck
        return leaf

    # get attribute A that best splits S
    A = MajorityError(S, Attributes)

    #create a root node for tree
    root = Node()
    root.splitsOn = A

    for v in values(S, A):
        #temp = Node(v)
        #root.add_child(temp)
        Sv = getSv(S, A, v)

        if len(Sv) == 0:
            # add leaf node w/ most common value of Label in S
            labels = {"": 0}
            for s in S:
                label = s["Label"]
                if label not in labels:
                    labels[label] = 0
                labels[label] += 1
        else:
            tempAttr = list(Attributes)
            tempAttr.remove(A)
            subtree = ID3(Sv, tempAttr, Label)
            #subtree.add_parent(root)
            subtree.label = v
            root.children.append(subtree)

    return root

def getSv(S, A, v):
    """Gets a list of every s in S where s[A] has value v"""
    return [s for s in S if s[A] == v]


def values(S, A):
    """gets every value that attribute A takes in S"""
    vals = set()
    for exm in S:
        vals.add(exm[A])

    return vals


def MajorityError(S, Attributes):
    """gets the attribute with the lowest majority error"""
    ME = majerrcalc(S)

    maxGain = 0
    maxAttr = ""
    for A in Attributes:
        gain = mehelper(S, A, ME)
        if gain > maxGain:
            maxGain = gain
            maxAttr = A

    return maxAttr


def mehelper(S, A, ME):
    """"""
    newME = 0.0
    for v in values(S, A):
        # creates a list of examples where attr A has value v
        Sv = getSv(S, A, v)
        ratio = len(Sv)/float(len(S))
        me = majerrcalc(Sv)
        newME += ratio * me

    return ME - newME

def majerrcalc(S):
    """gets the majority error of S"""
    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0
        labels[label] += 1

    maxLabel = ""
    numLabel = 0
    for label, num in labels.items():
        if num > numLabel:
            numLabel = num
            maxLabel = label

    numWrong = 0
    for s in S:
        label = s["Label"]
        if label != maxLabel:
            numWrong += 1

    return numWrong/float(len(S))

def infogain(S, Attributes):
    """gets the attribute with the best information gain"""
    entropy = entropycalc(S)

    maxInfo = 0
    maxAttr = ""
    for A in Attributes:
        infoGain = infohelper(S, A, entropy)
        if infoGain > maxInfo:
            maxInfo = infoGain
            maxAttr = A

    return maxAttr


def entropycalc(S):
    """calculates the entropy of S"""
    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0
        labels[label] += 1

    entropy = 0.0
    norm = len(S)  # normalizing value
    for (label, quant) in labels.items():
        ratio = quant/float(norm)
        entropy -= math.log((ratio), 2) * (ratio)

    return entropy


def infohelper(S, A, entropy):
    """calculates the info gain of a specific attribute"""
    newEnt = 0.0
    for v in values(S, A):
        # creates a list of examples where attr A has value v
        Sv = getSv(S, A, v)
        ratio = len(Sv)/float(len(S))
        ent = entropycalc(Sv)
        newEnt += ratio * ent

    return entropy - newEnt


file = open("input.txt")
Attributes = file.readline().strip().split(' ')

S = []
for line in file:
    example = {}
    i = 0
    for state in line.strip().split(' '):
        example[Attributes[i]] = state
        i += 1
    S.append(example)

Attributes.remove("Label")

Label = "0"

MegaRoot = ID3(S, Attributes, Label)

print("Hello World")
