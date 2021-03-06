import math

class Node:
    children = list()
    label = ""
    splitsOn = ""
    prediction = ""

    def __init__(self):
        self.children = list()
        self.label = ""
        self.splitsOn = ""
        self.prediction = ""


def ID3_Train(S, Attributes, Split, MaxDepth=None):
    splitFunc = None
    if Split == "InfoGain":
        splitFunc = InfoGain
    elif Split == "MajorityError":
        splitFunc = MajorityError
    elif Split == "GiniIndex":
        splitFunc = GiniIndex
    else:
        raise ValueError("Invalid Split function passed to ID3")

    return id3_main(S, Attributes, splitFunc, MaxDepth, 0)


def id3_main(S, Attributes, Split, MaxDepth, depth):
    # check if max depth reached
    if depth == MaxDepth:
        return mostCommonLabelLeaf(S)

    # check if all labels are the same
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

    # check if all attributes have been used
    if len(Attributes) == 0:
        return mostCommonLabelLeaf(S)

    # get attribute A that best splits S
    A = Split(S, Attributes)

    #create a root node for tree
    root = Node()
    root.splitsOn = A

    for v in Attributes[A]:
        Sv = getSv(S, A, v)

        if len(Sv) == 0:
            # add leaf node w/ most common value of Label in S
            leaf = mostCommonLabelLeaf(S, v)
            root.children.append(leaf)
        else:
            tempAttr = dict(Attributes)
            tempAttr.pop(A)
            subtree = id3_main(Sv, tempAttr, Split, MaxDepth, depth+1)
            subtree.label = v
            root.children.append(subtree)

    return root


def mostCommonLabelLeaf(S, v=None):
    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0
        labels[label] += 1

    maxNum = 0
    maxLabel = ""
    for label, num in labels.items():
        if num > maxNum:
            maxNum = num
            maxLabel = label

    leaf = Node()
    leaf.prediction = maxLabel
    leaf.label = v
    return leaf


def getSv(S, A, v):
    """Gets a list of every s in S where s[A] has value v"""
    return [s for s in S if s[A] == v]


def values(Attributes, A):
    """gets every value that attribute A takes"""
    return Attributes[A]


def MajorityError(S, Attributes):
    """gets the attribute with the lowest majority error"""
    ME = ME_calc(S)

    maxGain = -1  # in case purity gain is 0 across the board
    maxAttr = ""
    for A in Attributes:
        gain = ME_helper(S, Attributes, A, ME)
        if gain > maxGain:
            maxGain = gain
            maxAttr = A

    return maxAttr


def ME_helper(S, Attributes, A, ME):
    """"""
    newME = 0.0
    for v in Attributes[A]:
        # creates a list of examples where attr A has value v
        Sv = getSv(S, A, v)
        ratio = len(Sv)/float(len(S))
        me = ME_calc(Sv)
        newME += ratio * me

    return ME - newME


def ME_calc(S):
    """gets the majority error of S"""
    if len(S) == 0:
        return 0

    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0
        labels[label] += 1

    numLabel = 0
    for label, num in labels.items():
        if num > numLabel:
            numLabel = num

    return 1 - numLabel/float(len(S))


def InfoGain(S, Attributes):
    """gets the attribute with the best information gain"""
    entropy = ent_calc(S)

    maxInfo = -1
    maxAttr = ""
    for A in Attributes:
        infoGain = infohelper(S, Attributes, A, entropy)
        if infoGain > maxInfo:
            maxInfo = infoGain
            maxAttr = A

    return maxAttr


def infohelper(S, Attributes, A, entropy):
    """calculates the info gain of a specific attribute"""
    newEnt = 0.0
    for v in Attributes[A]:
        # creates a list of examples where attr A has value v
        Sv = getSv(S, A, v)
        ratio = len(Sv)/float(len(S))
        ent = ent_calc(Sv)
        newEnt += ratio * ent

    return entropy - newEnt


def ent_calc(S):
    """calculates the entropy of S"""
    if len(S) == 0:
        return 0

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


def GiniIndex(S, Attributes):
    """gets the attribute with the lowest Gini Index"""
    initialGini = GI_calc(S)

    maxInfo = -1
    maxAttr = ""
    for A in Attributes:
        infoGain = ginihelper(S, Attributes, A, initialGini)
        if infoGain > maxInfo:
            maxInfo = infoGain
            maxAttr = A

    return maxAttr


def ginihelper(S, Attributes, A, gini):
    """calculates the new gini index of splitting on a specific attribute"""
    newGI = 0.0
    for v in Attributes[A]:
        # creates a list of examples where attr A has value v
        Sv = getSv(S, A, v)
        ratio = len(Sv)/float(len(S))
        gi = GI_calc(Sv)
        newGI += ratio * gi

    return gini - newGI


def GI_calc(S):
    """calculates the Gini Index of S"""
    if len(S) == 0:
        return 0

    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0
        labels[label] += 1

    gini = 1.0
    norm = len(S)  # normalizing value
    for (label, quant) in labels.items():
        ratio = quant/float(norm)
        gini -= ratio**2

    return gini


def ID3_Test(Tree, S):
    wrong = 0
    for s in S:
        label = get_label(s, Tree)
        if label != s["Label"]:
            wrong += 1
    return wrong/float(len(S))


def get_label(s, Tree):
    if Tree.prediction != "":
        return Tree.prediction
    
    newTree = None
    for node in Tree.children:
        if node.label == s[Tree.splitsOn]:
            newTree = node
            break

    return get_label(s, newTree)