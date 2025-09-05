from sklearn.datasets import load_digits
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import is_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import time
from sklearn.datasets import load_digits, fetch_openml
from sklearn.datasets import load_breast_cancer
from deepforest import CascadeForestClassifier
#import deepforest
#print("Loaded deepforest from:", deepforest.__file__)
from deepforest.cascade import CascadeForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_lfw_pairs
from sklearn.preprocessing import OrdinalEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedShuffleSplit


#my_data_folder = os.path.join(os.getcwd(), "my_openml_data")
#usps = fetch_openml('usps', version=1, parser='auto')
#X = usps.data.astype(np.float32)  # (9298, 256)
#y = usps.target.astype(np.int8)   # (9298,) digits 0-9

#X, y = fetch_openml(name="mammographic-mass", as_frame=True, return_X_y=True)
#y = y.astype(int)

#data = fetch_lfw_pairs(subset='train', resize=0.4, color=False)

#X_left = data.pairs[:, 0]
#X_right = data.pairs[:, 1]
#y = data.target
#X = np.array([np.hstack((l.ravel(), r.ravel())) for l, r in zip(X_left, X_right)])
'''
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
# Split FIRST to prevent leakage
data = np.hstack((X.values, y.values.reshape(-1, 1)))
'''

#data = pd.read_csv(r'C:\Users\David\PycharmProjects\Examensarbete\Datasets\kr-vs-kp.data', header=None)


#X = pd.DataFrame(data.data, columns=data.feature_names)
#y = pd.Series(data.target, name="target")
#data = np.hstack((X.values, y.values.reshape(-1, 1)))
#np.random.shuffle(data)

#df = pd.read_csv('/Users/hehexdddd/PycharmProjects/PythonProject/tic-tac-toe.data', header=None)

# Encode all categorical features
#label_encoders = []
#for col in df.columns:
    #le = LabelEncoder()
    #df[col] = le.fit_transform(df[col])
    #label_encoders.append(le)

# Separate features and target
#X = df.iloc[:, :-1].values
#y = df.iloc[:, -1].values
'''
df = pd.read_csv('/Users/hehexdddd/PycharmProjects/PythonProject/diabetes.csv', header=0)
print(df.head())
print(df.columns)
if 'class' in df.columns and df.columns[-1] != 'class':
    cols = [c for c in df.columns if c != 'class'] + ['class']
    df = df[cols]


df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

#enc = OrdinalEncoder(dtype=np.int64)
#df[:] = enc.fit_transform(df)
df = df.apply(pd.to_numeric)

data = df.to_numpy()
X, y = data[:, :-1], data[:, -1]

np.random.shuffle(data)
'''

'''
heart = fetch_ucirepo(id=45)
X = heart.data.features.copy()
y = heart.data.targets.copy()

# Coerce to numeric (turns '?' etc. into NaN)
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN across features or target
df = pd.concat([X, y], axis=1).dropna().reset_index(drop=True)

# Binarize target if it's 0..4 ("num"); keep as-is if already 0/1
target_col = y.columns[0]
if df[target_col].nunique() > 2 or df[target_col].max() > 1:
    df[target_col] = (df[target_col] > 0).astype(int)
else:
    df[target_col] = df[target_col].astype(int)

# --- OPTIONAL: min–max scale FEATURES (not the label) ---
X_np = df.drop(columns=[target_col]).to_numpy(dtype=np.float64)
X_min = X_np.min(axis=0)
X_max = X_np.max(axis=0)
den = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
X_np = (X_np - X_min) / den
# --------------------------------------------------------

y_np = df[target_col].to_numpy(dtype=np.int64)

# Your format: features first, label last
data = np.hstack([X_np, y_np.reshape(-1, 1)])
np.random.shuffle(data)

#data = df.to_numpy()

X_clean = df.drop(columns=[target_col])
y_clean = df[target_col]
'''

'''
d = load_digits()
X, y = d.data, d.target
#y = (y-y.min())/(y.max()-y.min())
data = np.column_stack([X, y])
rng = np.random.default_rng(42)
rng.shuffle(data, axis=0)
'''

'''
ms = fetch_openml('mushroom', version=1, as_frame=True)

# Encode target (poisonous=1, edible=0)
y_full = (ms.target == 'p').astype(int)

# Replace '?' with NaN, then one-hot encode all features, fill NaNs with 0
X_full = pd.get_dummies(ms.data.replace('?', np.nan)).fillna(0).astype(int)

# --- Stratified subset of 2000 samples ---
sss = StratifiedShuffleSplit(n_splits=1, test_size=len(X_full) - 2000, random_state=42)
keep_idx, _ = next(sss.split(X_full, y_full))
X_sub = X_full.iloc[keep_idx]
y_sub = y_full.iloc[keep_idx]

# --- Shuffle the subset and split ---
# Convert to numpy
X_np = X_sub.to_numpy()
y_np = y_sub.to_numpy().reshape(-1, 1)
#X_np = X.to_numpy()
#y_np = y.to_numpy()

#y_np = y_np.reshape(-1, 1)  # Makes it shape (n_samples, 1)

data = np.hstack([X_np, y_np])
np.random.shuffle(data)

# Final X, y split
X = data[:, :-1]
y = data[:, -1].astype(int)
'''
'''
bn = fetch_openml('banknote-authentication', version=1, as_frame=True)
X = bn.data
y = (bn.target.astype(int))  # '0'/'1' strings → ints

data = np.hstack([X.values, y.values.reshape(-1,1)])
np.random.shuffle(data)
'''
'''
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

data = np.hstack([X.values, y.values.reshape(-1,1)])
np.random.shuffle(data)
# Train/test split (your model expects this)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3
)
'''
'''
maternal_health_risk = fetch_ucirepo(id=863)

X = maternal_health_risk.data.features
y = maternal_health_risk.data.targets

le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

data = np.hstack([X.values, y_encoded.reshape(-1, 1)])
np.random.shuffle(data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3
)
'''

raisin = fetch_ucirepo(id=850)

X = raisin.data.features
y = raisin.data.targets

le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

data = np.hstack([X.values, y_encoded.reshape(-1, 1)])
np.random.shuffle(data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3
)

'''
ds = fetch_openml(data_id=36, as_frame=True)
df = ds.frame.dropna().reset_index(drop=True)

X = df.drop(columns=['class'])
y = df['class']

y_encoded = LabelEncoder().fit_transform(y)

data = np.hstack([X.to_numpy(dtype=np.float64), y_encoded.reshape(-1, 1)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3
)
'''
start_df = time.time()
model = CascadeForestClassifier(n_trees=100, n_estimators=4)
print("Type:", type(model))
#Testing Accuracy: 95.100 %
#3 layers, 100 trees, depth 10

#Testing Accuracy: 94.033 % # 2 estimators, 20 trees
#Testing Accuracy: 93.900 % # 4 estimators, 10 trees #mnist
#Testing Accuracy: 94.400 % # 1 estimators, 40 trees

print("Is classifier:", is_classifier(model))
print("Has _estimator_type:", hasattr(model, "_estimator_type"))
print("Value of _estimator_type:", getattr(model, "_estimator_type", "NOT FOUND"))


#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3
#)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
end_df = time.time()


print("\nTesting Accuracy: {:.3f} %".format(acc))
print(f"Total Time Deep Forest: {end_df - start_df:.4f} sec")
#Testing Accuracy: 90.767 %
#letter dataset Testing Accuracy: 87.850 %, 40 trees, 2 forest, 3 layers, depth 10
#pendigits dataset Testing Accuracy: 99.030 % 40 trees, 2 forest, 3 layers, depth 10
#mnist dataset Testing Accuracy: 94.400 % # 40 trees, 2 forest, 3 layers, depth 10
#usps dataset Testing Accuracy: 95.448 % # 40 trees, 2 forest, 3 layers, depth 10


start_dt = time.time()
dt = DecisionTreeClassifier(max_depth=30, min_samples_split = 10, min_impurity_decrease = 5e-3)

# Train the model
dt.fit(X_train, y_train)

# Make predictions
dty_train_pred = dt.predict(X_train)
dty_test_pred = dt.predict(X_test)

# Evaluate
dttrain_acc = accuracy_score(y_train, dty_train_pred)
dttest_acc = accuracy_score(y_test, dty_test_pred)

end_dt = time.time()
print(f"DT Train Accuracy: {dttrain_acc:.4f}")
print(f"DT Test Accuracy:  {dttest_acc:.4f}")
print(f"Total Time Decision Tree: {end_dt - start_dt:.4f} sec")

start_rf = time.time()
rf = RandomForestClassifier(n_estimators=60, max_depth=30, min_samples_split = 10, min_impurity_decrease = 5e-5)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
rfy_train_pred = model.predict(X_train)
rfy_test_pred = model.predict(X_test)

# Evaluate
rftrain_acc = accuracy_score(y_train, rfy_train_pred)
rftest_acc = accuracy_score(y_test, rfy_test_pred)
end_rf = time.time()
print(f"RF Train Accuracy: {rftrain_acc:.4f}")
print(f"RF Test Accuracy:  {rftest_acc:.4f}")
print(f"Total Time Random Forest: {end_rf - start_rf:.4f} sec")
#Testing Accuracy: 97.076 load_breast_cancer unrestricted DF 400 trees 4 forest.
#Testing Accuracy 98.246 load_breast_cancer unrestricted DF 800 trees 8 forest

#KRVSKP
#Testing Accuracy DF 400 trees 4 forest unrestricted 99.166 9 seconds
#Testing Accuracy DF 800 trees 8 forest unrestricted 99.166 26 seconds
#Testing Accuracy DT 0.9854 depth 30
#Testing Accuracy RF 0.9917 50 trees depth 30

#tictactoe
#Testing Accuracy DF 400 trees 4 forest unrestricted 98.611 19 seconds
#Testing Accuracy DF 800 trees 8 forest unrestricted 98.264 18 seconds
#Testing Accuracy DT 0.9062 depth 30
#Testing Accuracy RF 0.9861 50 trees depth 30



