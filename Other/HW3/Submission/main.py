from perceptron import *

bank_note = "bank-note/"

f = open(bank_note + "train.csv")

Attributes = ["Variance", "Skewness", "Curtosis", "Entropy", "Label"]

S_train = []
for line in f:
    attrs = line.strip().split(',')
    example = {}
    for i in range(0, len(Attributes)):
        if Attributes[i] == "Label":
            attrs[i] = 1 if attrs[i] == '1' else -1
        example[Attributes[i]] = attrs[i]
    S_train.append(example)

f = open(bank_note + "test.csv")
S_test = []
for line in f:
    attrs = line.strip().split(',')
    example = {}
    for i in range(0, len(Attributes)):
        if Attributes[i] == "Label":
            attrs[i] = 1 if attrs[i] == '1' else -1
        example[Attributes[i]] = attrs[i]
    S_test.append(example)

Attributes.remove("Label")

print "Standard Perceptron:"
w = Perceptron_Standard(S_train, Attributes, .1, 10)
err_train = Perceptron_Test(w, S_train, Attributes)
err_test = Perceptron_Test(w, S_test, Attributes)
print "w: " + str(w)
print "Training Error: " + str(err_train)
print "Testing Error: " + str(err_test)

print "\nVoted Perceptron:"
WC = Perceptron_Voted(S_train, Attributes, .1, 10)
err_train = Perceptron_Test_Voted(WC, S_train, Attributes)
err_test = Perceptron_Test_Voted(WC, S_test, Attributes)
print "See all_weights.txt for weight vectors"
o = open("all_weights.txt", "w")
o.write("Votes\t\tWeight Vector\n")
for w, c in zip(WC[0], WC[1]):
    o.write("votes = " + str(c) + ",\tw = " + str(w) + "\n")
print "Training Error: " + str(err_train)
print "Testing Error: " + str(err_test)

print "\nAverage Perceptron:"
w = Perceptron_Average(S_train, Attributes, .1, 10)
err_train = Perceptron_Test(w, S_train, Attributes)
err_test = Perceptron_Test(w, S_test, Attributes)
print "a: " + str(w)
print "Training Error: " + str(err_train)
print "Testing Error: " + str(err_test)