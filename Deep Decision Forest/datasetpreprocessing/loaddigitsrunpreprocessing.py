import heapq

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits
import Trees
import modify_data
import backpropagationShapley
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
#from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine

#from modify_data import print_train_score
from sklearn.datasets import load_digits
d = load_digits()
#preprocessing
X, y = d.data, d.target
#y = (y-y.min())/(y.max()-y.min())
data = np.column_stack([X, y])
rng = np.random.default_rng(42)
rng.shuffle(data, axis=0)

train_t, test = modify_data.train_test_split(data, 0.7)
train_t, val = modify_data.train_test_split(train_t, 0.8)

rng = np.random.RandomState(0)

train = train_t.copy()


possible_classes = np.unique(train[:,-1])

unique, counts = np.unique(data[:, -1], return_counts=True)
train_unique, train_count = np.unique(train[:, -1], return_counts=True)
test_unique, test_count = np.unique(test[:, -1], return_counts=True)
#end preprocessing
