# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np


## Import data ##

bank_note = "bank-note/"

train_data, train_labels, test_data, test_labels = [], [], [], []

f = open(bank_note + "train.csv")
for line in f:
    attrs = line.strip().split(',')
    train_data.append([float(s) for s in attrs[:-1]])
    if attrs[-1] == '1':
        train_labels.append(1)
    else:
        train_labels.append(0)

f = open(bank_note + "test.csv")
for line in f:
    attrs = line.strip().split(',')
    test_data.append([float(s) for s in attrs[:-1]])
    if attrs[-1] == '1':
        test_labels.append(1)
    else:
        test_labels.append(0)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

#f = open('output.txt', 'w')

## construct NN ##

widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
acts = [keras.activations.tanh, tf.nn.relu]
inits = [tf.contrib.layers.xavier_initializer(), tf.initializers.he_normal()]
for i in range(2):
    print('Act/Int ' + str(i))
    #f.write('Act/Int ' + str(i) + '\n')
    for w in widths:
        for d in depths:
            layers = [keras.layers.Dense(units=w, activation=acts[i], kernel_initializer=inits[i], bias_initializer=inits[i], input_dim=4)]
            layers.extend([keras.layers.Dense(units=w, activation=acts[i], kernel_initializer=inits[i], bias_initializer=inits[i]) for _ in range(d - 2)])
            layers.append(keras.layers.Dense(units=1, activation=acts[i], kernel_initializer=inits[i], bias_initializer=inits[i]))

            model = keras.Sequential(layers)

            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

            # Train model
            model.fit(train_data, train_labels, epochs=20, verbose=0)

            # Evaluate
            _, test_acc = model.evaluate(test_data, test_labels, verbose=0)
            _, train_acc = model.evaluate(train_data, train_labels, verbose=0)

            print(str(w) + '/' + str(d) + ': ' + str(round(train_acc, 3)) + ', ' + str(round(test_acc, 3)))
            #f.write(str(w) + '/' + str(d) + ': ' + str(round(train_acc, 3)) + ', ' + str(round(test_acc, 3)) + '\n')