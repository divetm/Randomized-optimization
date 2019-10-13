import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from mlrose import mlrose

# load our dataset
train_data = pd.read_csv("tumor_classification_data.csv", delimiter=";")

# extract the images and labels from the dictionary object
y = train_data.pop('malignant').values
ids = train_data.pop('id').values
X = train_data

# transform y into a column
y = y.T

# shuffle to avoid underlying distributions
X, y = shuffle(X, y, random_state=26)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)
X_train = scale(X_train)
X_test = scale(X_test)

nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
                                    algorithm='random_hill_climb', max_iters=1000,
                                    bias=True, is_classifier=True, learning_rate=0.0001,
                                    early_stopping=True, clip_max=5, max_attempts=100,
                                    random_state=3)
nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
                                   algorithm='simulated_annealing', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=0.0001,
                                   early_stopping=True, clip_max=5, max_attempts=100,
                                   random_state=3)
nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
                                   algorithm='genetic_alg', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=0.0001,
                                   early_stopping=True, clip_max=5, max_attempts=100,
                                   random_state=3)

neural_nets = [nn_model_ga, nn_model_rhc, nn_model_sa]

for nn in neural_nets:
    nn.fit(X_train, y_train)

    y_train_pred = nn.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train accuracy for {}: {}".format(nn, y_train_accuracy))

    y_test_pred = nn.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Train accuracy for {}: {}".format(nn, y_test_accuracy))
