import heapq

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits, fetch_openml
import Trees
import modify_data
import backpropagationShapley
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import os
#from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine

#from modify_data import print_train_score
from sklearn.datasets import load_digits
bn = fetch_openml('banknote-authentication', version=1, as_frame=True)
#Preprocessing
X = bn.data
y = (bn.target.astype(int))  # '0'/'1' strings → ints

data = np.hstack([X.values, y.values.reshape(-1,1)])
np.random.shuffle(data)


train_t, test = modify_data.train_test_split(data, 0.7)
train_t, val = modify_data.train_test_split(train_t, 0.8)
train = train_t.copy()

possible_classes = np.unique(train[:,-1])
unique, counts = np.unique(data[:, -1], return_counts=True)
train_unique, train_count = np.unique(train[:, -1], return_counts=True)
test_unique, test_count = np.unique(test[:, -1], return_counts=True)
