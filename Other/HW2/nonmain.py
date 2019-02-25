from id3 import id3_with_weight, render
### MAIN ###

attrList = ["1", "2", "3", "4", "Label"]

S = []
with open("logic_in.txt") as f:
    Attributes = {}

    line = f.readline()
    while line != "\n":
        splitLine = line.split(':')
        attr = splitLine[0]
        Attributes[attr] = "".join(splitLine[1].split()).split(',')
        line = f.readline()
    
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(' '):
            example[attrList[i]] = attr
            i += 1
        example["Weight"] = 1
        S.append(example)


tree = id3_with_weight(S, Attributes, None, 0)

render(tree, "4e_tree")
