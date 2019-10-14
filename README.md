# Randomized-optimization
*Assignment #2 - CS 7641 Machine Learning course - Charles Isbell & Michael Littman - Georgia Tech*

Please clone this git to a local project if you want to replicate the experiments reported in the assignment paper.

Virtual Environment
----
This project contains a virtual environment folder ```venv```. This folder contains all the files needed to create a virtual environment in which the project is supposed to run.

requirements.txt
----
This file contains all the necessary packages for this project. (Running ```pip install -r requirements.txt``` will install all the packages in your project's environment - should not be necessary if you are using the given ```venv```folder here)

The dataset
----
This dataset (```tumor_classification_data.csv```) is the dataset described in the first assignment paper "Supervised Learning". It can be downloaded from its original source:
* https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

mlrose package
----
This package is a clone from Andrew Rolling's take on the mlrose package originally written by Genevieve Hayes. It has been modified so it could fit the directory tree of this project (solving some import errors because of distinct relative paths). By cloning this repository into a local project, you will be able to use the package completely and directly! You can find the reference to both repositories (orginal and Andrew Rolling's fork) here:
- Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python. https://github.com/gkhayes/mlrose. Accessed: 13 oct. 2019.
- Rolling, A. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python. https://github.com/hiive/mlrose. Accessed: 13 oct. 2019.

This package allows to implement a number of Machine Learning, Randomized Optimization and SEarch algorithms. You can find its documentation here https://mlrose.readthedocs.io/.
Andrew Rolling's fork has solved some issues regarding Genetic Algorithms and sped up the MIMIC algorithm (using David S. Park's work - https://github.com/parkds/mlrose).

k_color_problem.py, four_peaks_problem.py and knapsack_problem.py
----
Each of these Python scripts represents the study of a particular optimization problem: k-color problem, four peaks problem and the knapsack problem. In each of these studies, we apply the 4 optimization algorithms to try and solve them. When ran, these scripts create csv files containing all the stats of their computations (fitness against iterations, attempts, the actual states considered...). It is important to modify the ```output_directory``` argument in these algorithms to set where these files should be saved.
From this files, we were able to extract all the important data showcased in the assignment paper (like graphs).


optimizing_weights_nn_tumor_classification_problem.py
----
This Python script recreates the conditions of the first assignment to study the Tumor classification problem with a Neural Network. Then, it applies the Random Hill Climbing, the Simulated annealing and a Genetic algorithm to optimize the weights of this Neural Netowork. When ran, the algorithm will perform these optimizations and return the training and testing and accuracy levels of the Neural Network after optimization of its weights by the different algorithms. It also returns the computational time needed by each algorithm to perform this optimization.
