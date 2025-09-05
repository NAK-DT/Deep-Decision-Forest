import heapq

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import Trees
import modify_data
import backpropagationShapley
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine

from modify_data import print_train_score
from sklearn.datasets import load_digits

#data = pd.read_csv('Datasets/mushroomfinal.csv')

start = datetime.now()
print("Start_time: ", start)

'''wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
y = wine_quality.data.targets'''

'''heart = fetch_ucirepo(id=45)
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
np.random.shuffle(data)'''


#data = pd.read_csv('C:\\Users\\hugos\\OneDrive\\Dokument\\Final_Thesis\\Examensarbete\\Datasets\\multiclasssyntheticredundancy6 (1).csv')

#data = pd.read_csv('Datasets/SynthDatasets/test.csv')
#data = np.array(data)
#np.random.shuffle(data)

#data = load_breast_cancer()
#data = load_wine()
#X = pd.DataFrame(data.data, columns=data.feature_names)
#y = pd.Series(data.target, name="target")
#y = (y-y.min())/(y.max()-y.min())
#data = pd.read_csv('C:\\Users\\David\\PycharmProjects\\Examensarbete\\Datasets\\kr-vs-kp.data', header=None)
#data = pd.read_csv(r'C:\Users\David\PycharmProjects\Examensarbete\Datasets\kr-vs-kp.data', header=None)


#X = pd.DataFrame(data.data, columns=data.feature_names)
#y = pd.Series(data.target, name="target")



#d = load_digits()
#X, y = d.data, d.target
#y = (y-y.min())/(y.max()-y.min())
#data = np.column_stack([X, y])
#rng = np.random.default_rng(42)
#rng.shuffle(data, axis=0)
#data = np.hstack((X.values, y.values.reshape(-1, 1)))
#np.random.shuffle(data)

#df = pd.read_csv(r'C:\Users\hugos\OneDrive\Dokument\Final_Thesis\Examensarbete\Datasets\chess+king+rook+vs+king+pawn\kr-vs-kp.data', header=None)
#df = pd.read_csv(r'C:\Users\hugos\OneDrive\Dokument\Final_Thesis\Examensarbete\Datasets\tic+tac+toe+endgame\tic-tac-toe.data', header=None)#
'''label_encoders = []
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders.append(le)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Convert to NumPy
data = df.to_numpy()'''

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

train_t, test = modify_data.train_test_split(data, 0.7)
train_t, val = modify_data.train_test_split(train_t, 0.8)

rng = np.random.RandomState(0)

train = train_t.copy()

#train[:,-1] = rng.permutation(train[:,-1])

possible_classes = np.unique(train[:,-1])
'''
#print(data[0])

#digits = load_digits()
#X = digits.data
#y = digits.target

#data = np.hstack((X, y.reshape(-1, 1)))
#np.random.shuffle(data)
#train, test = modify_data.train_test_split(data, 0.7)
'''

#my_data_folder = os.path.join(os.getcwd(), "my_openml_data")
#letter = fetch_openml('letter', version=1, as_frame=False)
#X = mnist['data']               # shape (70000, 784)
#y = mnist['target'].astype(int) # labels as int
#X = letter.data  # Shape: (20000, 16)
#y = letter.target  # Labels: 'A'-'Z' (strings)


#y = np.array([ord(char) - ord('A') for char in y])

#mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#X = mnist['data']               # shape (70000, 784)
# = mnist['target'].astype(int) # labels as int

# Optional: reduce dataset size for faster testing
#X = X[:10000]
#y = y[:10000]

# Combine into your format
#data = np.hstack((X, y.reshape(-1, 1)))
#np.random.shuffle(data)

# Split train/test
#train, test = modify_data.train_test_split(data, 0.7)
#Code for mutliple trees
unique, counts = np.unique(data[:, -1], return_counts=True)
train_unique, train_count = np.unique(train[:, -1], return_counts=True)
test_unique, test_count = np.unique(test[:, -1], return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))
print("Class distribution (train):", dict(zip(train_unique, train_count)))
print("Class distribution (test):", dict(zip(test_unique, test_count)))
layer1 = []
L1_data_holder = []
L1_preds = []
layer1_acc = []
all_layers = []
random_trees_included = True
random_trees = 33




Use_NoT3 = True
NoT_Multiplier = 1.5



K = 5
use_weights = False

#best so far weighted 40 trees 40 chosen features depth 100 number 2
#interesting runs 50 trees 50 features not weighted depth 100 number 1

NoT = 60
epochs = 70
Shapley_chosen_features = 50
Shapley_number_tries = 1
increase = 0

number_threshs = 3

start_features = Shapley_chosen_features
min_feats = 2
divided = 2
max_feats = 30

distribution = NoT-random_trees-1
#Code for testing re-training function for a single tree in layer 2
for i in range(NoT):
    #print(i)
    rand = Trees.DecisionTree()
    rand.random_train(train)
    layer1.append(rand)
    preds = rand.sum_predictions(test, False)
    L1_preds.append(preds)
    L1_data = rand.sum_predictions(train, True)
    L1_data_holder.append(L1_data)
L1_data_holder = np.array(L1_data_holder)

print("passed layer 1")

#print("prediction shapes: ", np.shape(temp_preds))
#print("training shapes: ", np.shape(temp))
#print("Mean accuracy layer 1 (pre retrain): ", layer1_acc.mean())
#print(layer1[0].base.left.left.parent.parent.information_gain == layer1[0].base.information_gain == layer1[0].base.right.parent.information_gain)
#print(layer1[0].base.parent)
L1_preds = np.array(L1_preds)
L1_data_holder = np.array(L1_data_holder.transpose())
new_data = modify_data.concat(L1_data_holder, train[:,-1])
#print("Data after changes: ", new_data)
new_preds = modify_data.concat(L1_preds.transpose(), test[:,-1])

#print("shape of new data: ", new_data.shape, "\n")
#print("shape of new preds: ", new_preds.shape, "\n")

print("total Data: ", np.sum(new_data[:,-1]==True))
layer2 = []
layer2_train_preds = []
layer2_test_preds = []
subset_size = int(0.6 * len(new_data))  # 80% subsampling of dataset size
#print("new_data", new_data.shape)
#print("subset_size", subset_size)

tree_subsets = []
tree_indices = []
for i in range(NoT):
    layer2_tree = Trees.DecisionTree()
    subset_indices = np.random.choice(len(new_data), size=subset_size, replace=True)
    df_subset = new_data[subset_indices]
    tree_subsets.append(df_subset)
    tree_indices.append(subset_indices)
    layer2_tree.random_train(df_subset, subset_indices)
    random_train_preds = layer2_tree.sum_predictions(new_data, True)
    layer2_train_preds.append(random_train_preds.copy())
    layer2.append(layer2_tree)


print("passed layer 2")



layer2_train_preds = np.array(layer2_train_preds)
layer2_train_preds = modify_data.concat(layer2_train_preds.transpose(), new_data[:,-1])


layer3 = []
layer3_tree_subsets = []
layer3_data = []
layer3_sub_inds = []
if (NoT > epochs) and Use_NoT3:
    NoT3 = int(epochs*NoT_Multiplier)
else:
    NoT3 = NoT
for i in range(NoT3):
    layer3_tree = Trees.DecisionTree()
    subset_indices2 = np.random.choice(len(layer2_train_preds), size=subset_size, replace=True)
    df_subset2 = layer2_train_preds[subset_indices2]
    tree_subsets.append(df_subset2)
    tree_indices.append(subset_indices2)
    layer3_tree.random_train(df_subset2, subset_indices2)
    random_train_preds = layer3_tree.sum_predictions(layer2_train_preds, True)
    layer3_data.append(random_train_preds.copy())
    layer3.append(layer3_tree)
    layer3_sub_inds.append(subset_indices2)
    layer3_tree_subsets.append(df_subset2)

layer3_data = np.array(layer3_data)
layer3_data = modify_data.concat(layer3_data.transpose(), layer2_train_preds[:,-1])
print("passed layer 3")

weights = modify_data.compute_oob_weights(layer3, layer3_sub_inds, layer2_train_preds)
if use_weights:
    MV_train = modify_data.Majority_voting_weighted(layer3_data, weights)
else:
    MV_train = modify_data.Majority_voting(layer3_data)
#print_train_score(MV_train, "(pretrain)")
#print("Major vote shape: ", MV_train[:,1].shape)
L3_train_accuracy = modify_data.score(MV_train[:,0], MV_train[:,1])
#MV_test = modify_data.Majority_voting(layer2_test_preds)
#L2_test_accuracy = layer2[0].score(MV_test[:,0], MV_test[:,1])

print("Layer 3 training accuraccy pre retrain: ", L3_train_accuracy)

accuracy = L3_train_accuracy

#print("Layer 2 validation accuracy pre retraining: ", L2_test_accuracy)

Data_sets = [train, new_data, layer2_train_preds, layer3_data]
Layers = [layer1, layer2, layer3]
#subsets = [tree_subsets[0], tree_subsets[1], layer3_sub_inds]

test_result = modify_data.test_predict(Layers, test, weights)
print("Layer 3 test accuracy pre retrain: ", test_result)
train_via_test = modify_data.test_predict(Layers, val, weights)
print("Validation via test path accuracy:", train_via_test)
test_accuracy = test_result

#print(subsets[0])
depth = len(Layers)
print(depth)
#print("train: ", len(train[:,0]))
average_differences = []

summed_differences = modify_data.all_layer_structure_differences(Layers)

for diff in summed_differences:
    average_differences.append([diff])

#print("Layer average differences:", summed_differences)

preds_prev_iter = None

if 'cooldown' not in globals():
    cooldown = {}
COOLDOWN_LEN = 3


iter_start = datetime.now()
for i in range(epochs):

    print(f"\n⏱️ Iteration {i + 1} starts at {iter_start.strftime('%H:%M:%S')}...")
    '''if epochs == 15:
        Shapley_chosen_features = 5
    elif epochs == 21:
        Shapley_chosen_features = 3'''
    print("Iteration", i+1, "Starts")

    trees = []

    for idx, tree in enumerate(Layers[-1]):
        eval_acc = modify_data.eval_invid_trees(tree, Data_sets[-2])

        trees.append([idx,eval_acc])

    for s in list(cooldown.keys()):
        cooldown[s] -= 1
        if cooldown[s] <= 0:
            cooldown.pop(s, None)

    eligible = [pair for pair in trees if cooldown.get(pair[0], 0) == 0]

    if len(eligible) >= K:
        worst = heapq.nsmallest(K, eligible, key=lambda x: x[1])
    else:
        worst = heapq.nsmallest(K, trees, key=lambda x: x[1])

    worst_tree_idx = [idx for idx, score in worst]
    print("Worst tree:", worst_tree_idx)

    # === Shapley ===
    print("Start shap single")
    ret_layerS_shap, ret_dataS_shap, best_testS_shap, best_featS_shap, accepted_count_S, accepted_slots_S, single_L3_inds, single_weights = (
        backpropagationShapley.find_best_improvement_single(
            Layers[-2:], Data_sets[-3], Data_sets[-2:], L3_train_accuracy,
            random_trees_included, distribution, Shapley_chosen_features, Shapley_number_tries, worst_tree_idx, number_threshs, possible_classes, deepcopy(layer3_sub_inds)
        )
    )

    print("Start shap propagate")
    ret_layerP_shap, ret_dataP_shap, best_testP_shap, best_featsP_shap, accepted_count_P, accepted_slots_P, propagate_L3_inds, propagate_weights = (
        backpropagationShapley.find_best_improvement_propagate(
            Layers, Data_sets, L3_train_accuracy, depth, [], [], [],
            random_trees_included, distribution,
            Shapley_chosen_features, [], Shapley_number_tries, [worst_tree_idx], number_threshs, possible_classes, deepcopy(layer3_sub_inds)
        )
    )

    #print("shap single feature: ", best_featS_shap)
    print("shap propagate feature: ", best_featsP_shap)


    if (best_testP_shap > best_testS_shap) & (best_testP_shap >= accuracy):
        print("Propagate > Singular")
        accepted_count = accepted_count_P
        if best_testP_shap == accuracy:
            propagate_layers = deepcopy(ret_layerP_shap)
            propagate_test_accuracy = modify_data.test_predict(propagate_layers, test, propagate_weights)


            if accuracy > test_accuracy:
                early_phase = False
            else:
                early_phase = (i < epochs/2)

            if (propagate_test_accuracy < test_accuracy) and not early_phase:
                print("original best")
                print(propagate_test_accuracy)
                print("accuracy post retrain (test data): ", test_accuracy)
                print("accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)

                # after MV_data computed
                if use_weights:
                    MV_data = modify_data.Majority_voting_weighted(Data_sets[-1], weights)
                else:
                    MV_data = modify_data.Majority_voting(Data_sets[-1])
                preds_iter = MV_data[:,0]
                acc_train = modify_data.score(MV_data[:, 0], MV_data[:, 1])
                maj_acc = max(np.mean(MV_data[:, 1] == c) for c in np.unique(MV_data[:, 1]))
                changed_pct = np.mean(preds_iter != preds_prev_iter) if epochs > 0 else np.nan
                print(
                    f"iter {i}: train_acc={acc_train:.3f}, maj_acc={maj_acc:.3f}, changed={changed_pct:.3%}, accepted_L3={accepted_count}")
                preds_prev_iter = preds_iter.copy()
                accepted_slots = []
                proposed = set(worst_tree_idx)
                accepted = set(accepted_slots)
                rejected = proposed - accepted
                iter_end = datetime.now()
                duration = iter_end - iter_start
                print(f"✅ Iteration {i + 1} ended at {iter_end.strftime('%H:%M:%S')} | Duration: {duration}")


                for s in rejected:
                    cooldown[s] = COOLDOWN_LEN

                for s in accepted:
                    cooldown.pop(s, None)
                if Shapley_chosen_features < max_feats:
                    Shapley_chosen_features = int(Shapley_chosen_features + increase)
                else:
                    Shapley_chosen_features = start_features
                continue
        layer3_sub_inds = propagate_L3_inds
        weights = propagate_weights
        accepted_slots = accepted_slots_P
        Layers = deepcopy(ret_layerP_shap)
        Data_sets[1:] = deepcopy(ret_dataP_shap)
        accuracy = best_testP_shap
        print("shap Propagate accuracy", accuracy)
        iter_end = datetime.now()
        duration = iter_end - iter_start
        print(f"✅ Iteration {i + 1} ended at {iter_end.strftime('%H:%M:%S')} | Duration1: {duration}")





    elif (best_testS_shap > best_testP_shap) & (best_testS_shap >= accuracy):
        print("Propagate < Singular")

        accepted_count = accepted_count_S
        if best_testS_shap == accuracy:
            single_layers = deepcopy(Layers)
            single_layers[-2:] = deepcopy(ret_layerS_shap)
            single_test_accuracy = modify_data.test_predict(single_layers, test, single_weights)
            if accuracy > test_accuracy:
                early_phase = False
            else:
                early_phase = (i < epochs/2)
            if (single_test_accuracy < test_accuracy) and not early_phase:
                print("original best")
                print(single_test_accuracy)
                print("accuracy post retrain (test data): ", test_accuracy)
                print("accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)
                if use_weights:
                    MV_data = modify_data.Majority_voting_weighted(Data_sets[-1], weights)
                else:
                    MV_data = modify_data.Majority_voting(Data_sets[-1])
                preds_iter = MV_data[:, 0]
                acc_train = modify_data.score(MV_data[:, 0], MV_data[:, 1])
                maj_acc = max(np.mean(MV_data[:, 1] == c) for c in np.unique(MV_data[:, 1]))
                changed_pct = np.mean(preds_iter != preds_prev_iter) if epochs > 0 else np.nan
                print(
                    f"iter {i}: train_acc={acc_train:.3f}, maj_acc={maj_acc:.3f}, changed={changed_pct:.3%}, accepted_L3={accepted_count}")
                preds_prev_iter = preds_iter.copy()
                accepted_slots = []
                proposed = set(worst_tree_idx)
                accepted = set(accepted_slots)
                rejected = proposed - accepted
                iter_end = datetime.now()
                duration = iter_end - iter_start
                print(f"✅ Iteration {i + 1} ended at {iter_end.strftime('%H:%M:%S')} | Duration: {duration}")


                for s in rejected:
                    cooldown[s] = COOLDOWN_LEN  # start cooldown for non-improving slots

                for s in accepted:
                    cooldown.pop(s, None)

                if Shapley_chosen_features < max_feats:
                    Shapley_chosen_features = int(Shapley_chosen_features + increase)
                else:
                    Shapley_chosen_features = start_features
                continue

        layer3_sub_inds = single_L3_inds
        weights = single_weights
        accepted_slots = accepted_slots_S
        Layers[-2:] = deepcopy(ret_layerS_shap)
        Data_sets[-2:] = deepcopy(ret_dataS_shap)
        accuracy = best_testS_shap
        print("shap Single accuracy", accuracy)


    elif (best_testS_shap == best_testP_shap) & (best_testS_shap >= accuracy):
        print("Propagate == Singular")
        single_layers = deepcopy(Layers)
        single_layers[-2:] = deepcopy(ret_layerS_shap)
        propagate_layers = deepcopy(ret_layerP_shap)
        single_test_accuracy = modify_data.test_predict(single_layers, test, single_weights)
        val_single = modify_data.test_predict(single_layers, val, weights)
        propagate_test_accuracy = modify_data.test_predict(propagate_layers, test, propagate_weights)
        val_propagate = modify_data.test_predict(propagate_layers, val, weights)
        randomDouble = np.random.rand()
        test_improved = single_test_accuracy > test_accuracy
        train_improved = best_testS_shap > accuracy
        propagate_test_improved = propagate_test_accuracy > test_accuracy
        propagate_train_improved = best_testP_shap > accuracy

        # Always accept if test/train improved in early epochs
        if accuracy > test_accuracy:
            early_phase = False
        else:
            early_phase = i < epochs / 2

        print("prop:", propagate_test_accuracy, " sing:", single_test_accuracy)
        print(f"val prop: {val_propagate}, val sing: {val_single}")
        if (single_test_accuracy > propagate_test_accuracy or early_phase) and (test_improved or (train_improved and randomDouble <= 0.5)):
            # Do update
            print("Singular was better")
            layer3_sub_inds = single_L3_inds
            weights = single_weights
            accepted_count = accepted_count_S
            accepted_slots = accepted_slots_S
            Layers[-2:] = deepcopy(ret_layerS_shap)
            Data_sets[-2:] = deepcopy(ret_dataS_shap)
            accuracy = best_testS_shap

        elif (single_test_accuracy <= propagate_test_accuracy or early_phase) and (propagate_test_improved or propagate_train_improved):
            print("Propagate was better")
            layer3_sub_inds = propagate_L3_inds
            weights = propagate_weights
            accepted_count = accepted_count_P
            accepted_slots = accepted_slots_P
            Layers = deepcopy(ret_layerP_shap)
            Data_sets[1:] = deepcopy(ret_dataP_shap)
            accuracy = best_testP_shap
        else:
            print("Original best")
            accepted_count = accepted_count_S
            print("accuracy post retrain (test data): ", test_accuracy)
            print("accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)
            summed_differences = modify_data.all_layer_structure_differences(Layers)
            for k in range(len(summed_differences)):
                average_differences[k].append(summed_differences[k])
            if use_weights:
                MV_data = modify_data.Majority_voting_weighted(Data_sets[-1], weights)
            else:
                MV_data = modify_data.Majority_voting(Data_sets[-1])
            preds_iter = MV_data[:, 0]
            acc_train = modify_data.score(MV_data[:, 0], MV_data[:, 1])
            maj_acc = max(np.mean(MV_data[:, 1] == c) for c in np.unique(MV_data[:, 1]))
            changed_pct = np.mean(preds_iter != preds_prev_iter) if epochs > 0 else np.nan
            print(
                f"iter {i}: train_acc={acc_train:.3f}, maj_acc={maj_acc:.3f}, changed={changed_pct:.3%}, accepted_L3={accepted_count}")
            preds_prev_iter = preds_iter.copy()
            accepted_slots = []
            proposed = set(worst_tree_idx)
            accepted = set(accepted_slots)
            rejected = proposed - accepted
            iter_end = datetime.now()
            duration = iter_end - iter_start
            print(f"✅ Iteration {i + 1} ended at {iter_end.strftime('%H:%M:%S')} | Duration3: {duration}")

            for s in rejected:
                cooldown[s] = COOLDOWN_LEN  # start cooldown for non-improving slots

            for s in accepted:
                cooldown.pop(s, None)

            if Shapley_chosen_features < max_feats:
                Shapley_chosen_features = int(Shapley_chosen_features + increase)
            else:
                Shapley_chosen_features = start_features
            continue




    else:
        accepted_count = 0
        print("Original best")
        print("accuracy post retrain (test data): ", test_accuracy)
        print("accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)
        summed_differences = modify_data.all_layer_structure_differences(Layers)
        for k in range(len(summed_differences)):
            average_differences[k].append(summed_differences[k])
        if use_weights:
            MV_data = modify_data.Majority_voting_weighted(Data_sets[-1], weights)
        else:
            MV_data = modify_data.Majority_voting(Data_sets[-1])
        preds_iter = MV_data[:, 0]
        acc_train = modify_data.score(MV_data[:, 0], MV_data[:, 1])
        maj_acc = max(np.mean(MV_data[:, 1] == c) for c in np.unique(MV_data[:, 1]))
        changed_pct = np.mean(preds_iter != preds_prev_iter) if epochs > 0 else np.nan
        print(
            f"iter {i}: train_acc={acc_train:.3f}, maj_acc={maj_acc:.3f}, changed={changed_pct:.3%}, accepted_L3={accepted_count}")
        preds_prev_iter = preds_iter.copy()
        accepted_slots = []
        proposed = set(worst_tree_idx)
        accepted = set(accepted_slots)
        rejected = proposed - accepted
        iter_end = datetime.now()
        duration = iter_end - iter_start
        print(f"✅ Iteration {i + 1} ended at {iter_end.strftime('%H:%M:%S')} | Duration4: {duration}")

        for s in rejected:
            cooldown[s] = COOLDOWN_LEN  # start cooldown for non-improving slots

        for s in accepted:
            cooldown.pop(s, None)

        if Shapley_chosen_features < max_feats:
            Shapley_chosen_features = int(Shapley_chosen_features + increase)
        else:
            Shapley_chosen_features = start_features
        continue


    if use_weights:
        MV_data = modify_data.Majority_voting_weighted(Data_sets[-1], weights)
    else:
        MV_data = modify_data.Majority_voting(Data_sets[-1])
    preds_iter = MV_data[:, 0]
    acc_train = modify_data.score(MV_data[:, 0], MV_data[:, 1])
    maj_acc = max(np.mean(MV_data[:, 1] == c) for c in np.unique(MV_data[:, 1]))
    changed_pct = np.mean(preds_iter != preds_prev_iter) if epochs > 0 else np.nan
    print(
        f"iter {i}: train_acc={acc_train:.3f}, maj_acc={maj_acc:.3f}, changed={changed_pct:.3%}, accepted_L3={accepted_count}")
    preds_prev_iter = preds_iter.copy()

    proposed = set(worst_tree_idx)
    accepted = set(accepted_slots)
    rejected = proposed - accepted

    for s in rejected:
        cooldown[s] = COOLDOWN_LEN  # start cooldown for non-improving slots

    for s in accepted:
        cooldown.pop(s, None)

    test_accuracy = modify_data.test_predict(Layers, test, weights)
    train_via_test = modify_data.test_predict(Layers, val, weights)
    print("Validation via test path accuracy:", train_via_test)
    print("accuracy post retrain (test data): ", test_accuracy)
    print("accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)
    #ret_dataS_shap = np.array(ret_dataS_shap)
    #print(Layers)
    summed_differences = modify_data.all_layer_structure_differences(Layers)
    for k in range(len(summed_differences)):
        average_differences[k].append(summed_differences[k])
    #print("Layer average differences:", summed_differences)

    iter_end = datetime.now()
    duration = iter_end - iter_start
    print(f"✅ Iteration {i + 1} ended at {iter_end.strftime('%H:%M:%S')} | Duration6: {duration}")

#print(average_differences)




end = datetime.now()
print("End Time: ", end)
print(f"Duration: {end - start}")
print("Final done")