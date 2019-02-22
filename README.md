This is a machine learning library developed by William Frank for CS5350/6350 in University of Utah

## ID3
ID3_Train(S, Attributes, Split, MaxDepth=None)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
Split - The algorithm used to split the dataset, can be "InfoGain", "MajorityError", or "GiniIndex"
Maxdepth - The maximum depth of the tree
Returns - A tree structure where each non-leaf node contains the attribute that was split on, and each non-root node contains the value of the attribute the parent node split on. Leaf nodes contain the appropriate label for their data.

ID3_Test(Tree, S)
S - A list of dictionaries in the same form that ID3_Train takes
Tree - a tree output from ID3_Train
Returns - The classification error of S on Tree
