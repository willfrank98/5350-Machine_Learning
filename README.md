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


## Adaboost
AdaBoost_Train(S, Attributes, T)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
T - The number of iterations to perform
Returns - a (trees, alphas) tuple, a list of decision stumps and their respective weights

AdaBoost_Test(Hypothesis, S)
Hypothesis - The value that AdaBoost_Train returns, a (trees, alphas) tuple
S - A list of dictionaries in the same form that Adaboost_Train takes
Returns - The classification error of S on Hypothesis


## Bagging
Bagging_Train(S, Attributes, T)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
T - The number of iterations to perform
Returns - a list of decision trees to total votes from

Bagging_Test(Hypothesis, S)
Hypothesis - a list of decision trees, as returned by Bagging_Train
S - A list of dictionaries in the same form that Bagging_Train takes
Returns - The classification error of S on Hypothesis


## Random Forest
Random_Forest_Train(S, Attributes, T, num_features)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
T - The number of iterations to perform
num_features - 
Returns - a list of decision trees to total votes from

Random_Forest_Test(Hypothesis, S)
Hypothesis - a list of decision trees, as returned by Bagging_Train
S - A list of dictionaries in the same form that Bagging_Train takes
Returns - The classification error of S on Hypothesis


## Batch Gradient Descent
Batch_LMS(S, Attributes, R, Convergence)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
R - Learning rate
Covnergence - When any weight changes by at most Convergence the weights are returned
Returns - A list of weights of length len(Attributes)

## Stochastic Gradient Descent
Stochastic_LMS(S, Attributes, R, Convergence)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
R - Learning rate
Covnergence - When any weight changes by at most Convergence the weights are returned
Returns - A list of weights of length len(Attributes)

## Standard Perceptron
Perceptron_Standard(S, Attributes, R, MaxEpochs)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
R - Learning rate
MaxEpochs - Maximum number of iterations to perform
Returns - A list of weights of length len(Attributes)

## Averaged Perceptron
Perceptron_Average(S, Attributes, R, MaxEpochs)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
R - Learning rate
MaxEpochs - Maximum number of iterations to perform
Returns - A list of weights of length len(Attributes)

## Voted Perceptron
Perceptron_Voted(S, Attributes, R, MaxEpochs)
S - A list of dictionaries, where each dictionary is a training example of form Attribute:value for each attribute in Attributes, and "Label":"value"
Attributes - A dictionary of all attributes and their possible values, in the form "Attribute":["all", "possible", "values"], or "Attribute":["(numeric)"] for numerical attributes
R - Learning rate
MaxEpochs - Maximum number of iterations to perform
Returns - A list of weights of length len(Attributes), and a list of counts of the same length
