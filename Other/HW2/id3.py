import math
import random
#from graphviz import Graph


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

def id3_with_weight(S, Attributes, MaxDepth, depth):
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
    A = InfoGain(S, Attributes)

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
            subtree = id3_with_weight(Sv, tempAttr, MaxDepth, depth+1)
            subtree.label = v
            root.children.append(subtree)

    return root

def id3_rand_learn(S, Attributes, NumFeatures):
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
    A = InfoGain_rand(S, Attributes, NumFeatures)

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
            subtree = id3_rand_learn(Sv, tempAttr, NumFeatures)
            subtree.label = v
            root.children.append(subtree)

    return root


def mostCommonLabelLeaf(S, v=None):
    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0.0
        labels[label] += s["Weight"]

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
        ratio = get_len(Sv)/float(get_len(S))
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
            labels[label] = 0.0
        labels[label] += s["Weight"]

    entropy = 0.0
    norm = get_len(S)  # normalizing value
    for (label, quant) in labels.items():
        ratio = quant/float(norm)
        entropy -= math.log((ratio), 2) * (ratio)

    return entropy

def InfoGain_rand(S, Attributes, NumFeatures):
    """gets the attribute with the best information gain"""
    entropy = ent_calc(S)

    newAttrs = []
    cpyAttrs = list(Attributes)
    for _ in range(0, NumFeatures):
        if len(cpyAttrs) == 0:
            break
        rand = random.randint(0, len(cpyAttrs)-1)
        newAttrs.append(cpyAttrs[rand])
        del cpyAttrs[rand]

    maxInfo = -1
    maxAttr = ""
    for A in newAttrs:
        infoGain = infohelper(S, Attributes, A, entropy)
        if infoGain > maxInfo:
            maxInfo = infoGain
            maxAttr = A

    return maxAttr

def get_len(S):
    length = 0
    for s in S:
        length += s["Weight"]
    return length


def id3_weighted_err(Tree, S):
    wrong = 0.0
    for s in S:
        label = get_label(s, Tree)
        if label != s["Label"]:
            wrong += s["Weight"]
    return wrong


def get_label(s, Tree):
    if Tree.prediction != "":
        return Tree.prediction
    
    newTree = None
    for node in Tree.children:
        if node.label == s[Tree.splitsOn]:
            newTree = node
            break

    return get_label(s, newTree)


# def render(root, name):
#     graph = Graph()
#     nodes = get_nodes(root)
#     edges = get_edges(root)

#     for node in nodes:
#         graph.node(node[0], node[1])

#     for edge in edges:
#         graph.edge(edge[0], edge[1], edge[2])

#     graph.render(name, view=True, format="pdf")


# def get_nodes(root):
#     nodes = []
#     if root.splitsOn == "":
#         nodes.append((str(id(root)), "Label: " + root.prediction))
#     else:
#         nodes.append((str(id(root)), root.splitsOn))
    
#     for child in root.children:
#         nodes.extend(get_nodes(child))
#     return nodes


# def get_edges(root):
#     edges = []
#     for child in root.children:
#         edges.append((str(id(root)), str(id(child)), child.label))
#         edges.extend(get_edges(child))
#     return edges