import heapq

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from Tran import Trees
from Tran import modify_data
from Tran import backpropagationShapley
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
#from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine

#from modify_data import print_train_score
from sklearn.datasets import load_digits

df = pd.read_csv('C:/Users/David/PycharmProjects/Examensarbete/testsets/diabetes.csv')
if 'class' in df.columns and df.columns[-1] != 'class':
    cols = [c for c in df.columns if c != 'class'] + ['class']
    df = df[cols]

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df = df.apply(pd.to_numeric)
data = df.to_numpy()
np.random.shuffle(data)
train_t, test = modify_data.train_test_split(data, 0.7)
train_t, val = modify_data.train_test_split(train_t, 0.8)


train = train_t.copy()


possible_classes = np.unique(train[:,-1])
unique, counts = np.unique(data[:, -1], return_counts=True)
train_unique, train_count = np.unique(train[:, -1], return_counts=True)
test_unique, test_count = np.unique(test[:, -1], return_counts=True)
