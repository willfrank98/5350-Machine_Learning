import math
from graphviz import Graph


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


def ID3(S, Attributes, Label, Split):
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
    A = Split(S, Attributes)

    #create a root node for tree
    root = Node()
    root.splitsOn = A

    for v in Attributes[A]:
        Sv = getSv(S, A, v)

        if len(Sv) == 0:
            # add leaf node w/ most common value of Label in S
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
            root.children.append(leaf)
        else:
            tempAttr = dict(Attributes)
            tempAttr.pop(A)
            subtree = ID3(Sv, tempAttr, Label, Split)
            subtree.label = v
            root.children.append(subtree)

    return root

def getSv(S, A, v):
    """Gets a list of every s in S where s[A] has value v"""
    return [s for s in S if s[A] == v]


def values(Attributes, A):
    """gets every value that attribute A takes"""
    return Attributes[A]


def MajorityError(S, Attributes):
    """gets the attribute with the lowest majority error"""
    ME = ME_calc(S)

    maxGain = -1 # in case purity gain is 0 across the board
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

def InfoGain(S, Attributes):
    """gets the attribute with the best information gain"""
    entropy = ent_calc(S)

    maxInfo = 0
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
    print "Gini Index = " + "{:.2f}".format(initialGini)

    maxInfo = 0
    maxAttr = ""
    for A in Attributes:
        print "GI for " + A + " = "
        infoGain = ginihelper(S, Attributes, A, initialGini)
        if infoGain > maxInfo:
            maxInfo = infoGain
            maxAttr = A

    return maxAttr

def ginihelper(S, Attributes, A, gini):
    """calculates the new gini index of splitting on a specific attribute"""
    newGI = 0.0
    output = "$" + "{:.2f}".format(gini) + " - ("
    for v in Attributes[A]:
        # creates a list of examples where attr A has value v
        Sv = getSv(S, A, v)
        ratio = len(Sv)/float(len(S))
        gi = GI_calc(Sv)
        output += "\\frac{" + str(len(Sv)) + "}{" + str(len(S)) + "}(" + "{:.2f}".format(gi) + ") + "
        newGI += ratio * gi

    output += ") \\approx " + "{:.2f}".format(gini - newGI) + "$"
    print output
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


def render(root, name):
    graph = Graph()
    nodes = get_nodes(root)
    edges = get_edges(root)

    for node in nodes:
        graph.node(node[0], node[1])

    for edge in edges:
        graph.edge(edge[0], edge[1], edge[2])

    graph.render(name, view=True, format="pdf")


def get_nodes(root):
    nodes = []
    if root.splitsOn == "":
        nodes.append((str(id(root)), "Label: " + root.prediction))
    else:
        nodes.append((str(id(root)), root.splitsOn))
    
    for child in root.children:
        nodes.extend(get_nodes(child))
    return nodes


def get_edges(root):
    edges = []
    for child in root.children:
        edges.append((str(id(root)), str(id(child)), child.label))
        edges.extend(get_edges(child))
    return edges


file = open("tennis.txt")
numAttr = int(file.readline().strip())

listAttr = file.readline().strip().split(' ')
listAttr.remove("Label")
Attributes = {}
for i in range(0, numAttr):
    Attributes[listAttr[i]] = file.readline().strip().split(' ')
listAttr.append("Label") # Label isn't an attribute, but we need it to build examples

S = []
for line in file:
    example = {}
    i = 0
    for state in line.strip().split(' '):
        example[listAttr[i]] = state
        i += 1
    S.append(example)

Label = "0"

MegaRoot = ID3(S, Attributes, Label, InfoGain)

render(MegaRoot, None)
