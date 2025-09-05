import numpy as np
import pandas as pd
import Trees
import modify_data
import backpropagationShapley
import backpropagationExhaustive
from copy import deepcopy
from datetime import datetime
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

start = datetime.now()
print("Start_time: ", start)

'''
from collections import defaultdict

def get_sklearn_subtree_signature(tree, node_id):
    if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
        return ('leaf', tuple(tree.value[node_id][0]))
    left = get_sklearn_subtree_signature(tree, tree.children_left[node_id])
    right = get_sklearn_subtree_signature(tree, tree.children_right[node_id])
    return ('node', tree.feature[node_id], round(tree.threshold[node_id], 4), left, right)

def count_sklearn_subtree_redundancy(clf):
    tree = clf.tree_
    sig_counts = defaultdict(int)

    def traverse(node_id):
        sig = get_sklearn_subtree_signature(tree, node_id)
        sig_counts[sig] += 1
        if tree.children_left[node_id] != -1:
            traverse(tree.children_left[node_id])
        if tree.children_right[node_id] != -1:
            traverse(tree.children_right[node_id])

    traverse(0)

    duplicates = sum(1 for count in sig_counts.values() if count > 1)
    return {
        "total": sum(sig_counts.values()),
        "unique": len(sig_counts),
        "duplicates": duplicates,
        "redundancy_ratio": duplicates / len(sig_counts),
        "signatures": set(sig_counts)
    }

def get_subtree_signature(node):
    if node is None:
        return None
    if node.left is None and node.right is None:
        return ('leaf', node.value)
    left_sig = get_subtree_signature(node.left)
    right_sig = get_subtree_signature(node.right)
    return ('node', node.feature, round(node.threshold, 4), left_sig, right_sig)

def compare_ddf_to_sklearn(ddf_trees, sklearn_signatures):
    reused = 0
    ddf_signatures = set()

    for tree in ddf_trees:
        def collect_signatures(node):
            if node is None:
                return None
            sig = get_subtree_signature(node)
            ddf_signatures.add(sig)
            collect_signatures(node.left)
            collect_signatures(node.right)
        collect_signatures(tree.base)

    for sig in ddf_signatures:
        if sig in sklearn_signatures:
            reused += 1

    return {
        "ddf_total": len(ddf_signatures),
        "reused_from_sklearn": reused,
        "reuse_ratio": reused / len(ddf_signatures) if ddf_signatures else 0
    }


def generate_complex_boolean_function(n_features=10, max_expr_depth=3, min_positive_ratio=0.2, max_positive_ratio=0.8):
    features = [f'f{i+1}' for i in range(n_features)]

    def random_expr(depth=0):
        if depth >= max_expr_depth or (depth > 1 and random.random() < 0.3):
            var = random.choice(features)
            return var if random.random() < 0.5 else f'(not {var})'
        else:
            left = random_expr(depth + 1)
            right = random_expr(depth + 1)
            op = random.choice(['and', 'or'])
            return f'({left} {op} {right})'

    while True:
        top_exprs = [random_expr() for _ in range(random.randint(3, 6))]
        combine_op = random.choice(['and', 'or'])  # Make it complex at top
        final_expr = f' {combine_op} '.join(top_exprs)

        def boolean_function(row):
            env = {f'f{i+1}': bool(row[i]) for i in range(n_features)}
            return int(eval(final_expr, {}, env))

        # Generate and test balance
        input_combinations = np.array(np.meshgrid(*[[0, 1]] * n_features)).T.reshape(-1, n_features)
        y = np.apply_along_axis(boolean_function, 1, input_combinations)
        pos_ratio = np.mean(y)
        if min_positive_ratio <= pos_ratio <= max_positive_ratio:
            return boolean_function, final_expr

# === Dataset Creation ===
def create_dataset(func, n_features=10):
    input_combinations = np.array(np.meshgrid(*[[0, 1]] * n_features)).T.reshape(-1, n_features)
    X = pd.DataFrame(input_combinations, columns=[f'f{i+1}' for i in range(n_features)])
    y = np.apply_along_axis(func, 1, input_combinations)
    return X, y

# === Train and Plot ===

def evaluate_new_T(row):
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = row
    expr1 = f1 and (f2 or (f3 and f4))
    expr2 = (not f1) and (f5 or (f6 and f7))
    expr3 = f8 and (f9 or (f10 and f3))
    expr4 = (not f8) and (f2 or (f5 and f9))
    return int(expr1 or expr2 or expr3 or expr4)

#input_combinations = np.array(np.meshgrid(*[[0, 1]]*10)).T.reshape(-1, 10)
#columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']
#X = pd.DataFrame(input_combinations, columns=columns)
#y = np.apply_along_axis(evaluate_new_T, 1, input_combinations)
#data = np.hstack([input_combinations, y.reshape(-1, 1)])

func, expr = generate_complex_boolean_function()
print("Expression:", expr)

# Create the dataset
X, y = create_dataset(func)

# Convert to raw array format
data = np.hstack([X.values, y.reshape(-1, 1)])
'''

#data = pd.read_csv('Datasets/mushroomfinal.csv')
#data = pd.read_csv('Datasets/SynthDatasets/syntheticredundancy6.csv')
#data = pd.read_csv('Datasets/SynthDatasets/test.csv')
#print(data.info())
#data = np.array(data)
#np.random.shuffle(data)
#print("Data: ", data)
#train, test = modify_data.train_test_split(data, 0.7)

#Code for mutliple trees
#unique, counts = np.unique(data[:, -1], return_counts=True)
#print("Class distribution:", dict(zip(unique, counts)))
layer1 = []
L1_data_holder = []
L1_preds = []
data_indeces = []
layer1_acc = []
all_layers = []


random_trees_included = False

NoT = 5
random_trees = 33
Shapley_chosen_features = 2
Shapley_number_tries = 1
increase = 3


start_features = Shapley_chosen_features
min_feats = 2
divided = 2
max_feats = 30



distribution = NoT-random_trees-1
#Code for testing re-training function for a single tree in layer 2
from itertools import product

# === Generate Boolean truth table ===
'''
#report
def boolean_function(row):
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = row
    expr1 = f1 and (f2 or (f3 and f4))
    expr2 = (not f1) and (f5 or (f6 and f7))
    expr3 = f8 and (f9 or (f10 and f3))
    expr4 = (not f8) and (f2 or (f5 and f9))
    return int(expr1 or expr2 or expr3 or expr4)
'''



def boolean_function(row):
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = row
    expr1 = f4 and (f1 or (f2 and f5))
    expr2 = (not f3) and (f6 or (f7 and f8))
    expr3 = f9 and (f10 or (f1 and f7))
    expr4 = (not f2) and (f5 or (f8 and f6))
    return int(expr1 or expr2 or expr3 or expr4)




n_features = 10
X_full = np.array(list(product([0, 1], repeat=n_features)))
y_full = np.array([boolean_function(row) for row in X_full])
#data = np.hstack((X_full, y_full.reshape(-1, 1)))
truth_data = np.hstack([X_full, y_full.reshape(-1, 1)])
train = truth_data
test = truth_data
#train, test = modify_data.train_test_split(data, 0.7)
#np.random.shuffle(data)
# === Use your Trees.DecisionTree to train Layer 1 ===

clf_optimal = DecisionTreeClassifier(random_state=SEED, max_depth=10)
clf_optimal.fit(X_full, y_full)

# Create output directory if not already
os.makedirs("testtrees/optimal", exist_ok=True)

# Plot and save the optimal decision tree
plt.figure(figsize=(24, 12))
plot_tree(
    clf_optimal,
    filled=True,
    feature_names=[f'f{i+1}' for i in range(X_full.shape[1])],
    class_names=["0", "1"],
    rounded=True
)
plt.title("Optimal Decision Tree (Sklearn)")
plt.savefig("testtrees/optimal/tree.png")
plt.close()

for i in range(NoT):
    '''
    rand = Trees.DecisionTree()
    rand.random_train(train)
    layer1.append(rand)
    preds = rand.sum_predictions(test)
    L1_preds.append(preds)
    L1_data, data_index = modify_data.get_results(rand.base, [], [])
    L1_data_holder.append(L1_data)
    data_indeces.append(data_index)
    '''

    tree = Trees.DecisionTree()
    tree.train(truth_data)  # optimal training on full truth table
    layer1.append(tree)

    preds = tree.sum_predictions(truth_data[:, :-1])  # <- correct!
    L1_preds.append(preds)

    L1_data, data_index = modify_data.get_results(tree.base, [], [])
    L1_data_holder.append(L1_data)
    data_indeces.append(data_index)

data_indeces = np.array(data_indeces)
L1_data_holder = np.array(L1_data_holder)
print(L1_data_holder.shape, "   ", data_indeces.shape)
L1_data_holder, template_indeces, data_indeces = modify_data.rearange_data(L1_data_holder, data_indeces)
print(L1_data_holder.shape,"   ", data_indeces.shape)
L1_preds = np.array(L1_preds)
L1_data_holder = np.array(L1_data_holder)
new_data = modify_data.concat(L1_data_holder, train[:,-1])
#print("Data after changes: ", new_data)
new_preds = modify_data.concat(L1_preds.transpose(), test[:,-1])


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
    if (i > distribution) & random_trees_included:
        layer2_tree = Trees.DecisionTree()
        subset_indices = np.random.choice(len(new_data), size=subset_size, replace=True)
        df_subset = new_data[subset_indices]
        tree_subsets.append(df_subset)
        tree_indices.append(subset_indices)
        layer2_tree.random_train(df_subset, subset_indices)
        random_train_preds = layer2_tree.sum_predictions(new_data)
        layer2_train_preds.append(random_train_preds.copy())
        layer2.append(layer2_tree)

    else:

        layer2_tree = Trees.DecisionTree()

        #df_subset = new_data.sample(frac=0.8, random_state=i).to_numpy()  # assign 80% of the data to each tree.
        #subset_indices = df_subset.index.to_numpy()
        #print("length of new data", len(new_data))
        subset_indices = np.random.choice(len(new_data), size=subset_size, replace=True)
        df_subset = new_data[subset_indices]

        tree_subsets.append(df_subset)
        tree_indices.append(subset_indices)

        #print(f"Subset 1 shape: {tree_subsets[0].shape}")
        #print(f"Subset 1 indices shape: {tree_indices[0].shape}")


        layer2_tree.train(df_subset, subset_indices)
        final_preds = layer2_tree.sum_predictions(new_preds)
        #layer2_test_preds.append(final_preds)
        #accuracyL2 = layer2_tree.score(final_preds, new_preds[:, -1])
        train_preds = layer2_tree.sum_predictions(new_data)
        layer2_train_preds.append(train_preds.copy())
        #L2_train_accuracy = layer2_tree.score(train_preds, new_data[:, -1])
        layer2.append(layer2_tree)




#layer2_test_preds = np.array(layer2_test_preds)
#layer2_test_preds = modify_data.concat(layer2_test_preds.transpose(), new_preds[:,-1])
layer2_train_preds = np.array(layer2_train_preds)
layer2_train_preds = modify_data.concat(layer2_train_preds.transpose(), new_data[:,-1])

#MV_train = modify_data.Majority_voting(layer2_train_preds)
#print("Major vote shape: ", MV_train[:,1].shape)
#L3_train_accuracy = layer2[0].score(MV_train[:,0], MV_train[:,1])
#MV_test = modify_data.Majority_voting(layer2_test_preds)
#L2_test_accuracy = layer2[0].score(MV_test[:,0], MV_test[:,1])

#print("Layer 2 training accuraccy pre retrain: ", L2_train_accuracy)
#print("Layer 2 validation accuracy pre retraining: ", L2_test_accuracy)

layer3 = []
layer3_tree_subsets = []
layer3_data = []
layer3_sub_inds = []
for i in range(NoT):
    if (i > distribution) & random_trees_included:
        layer3_tree = Trees.DecisionTree()
        subset_indices2 = np.random.choice(len(layer2_train_preds), size=subset_size, replace=True)
        df_subset2 = layer2_train_preds[subset_indices2]
        tree_subsets.append(df_subset2)
        tree_indices.append(subset_indices2)
        layer3_tree.random_train(df_subset2, subset_indices2)
        random_train_preds = layer3_tree.sum_predictions(new_data)
        layer3_data.append(random_train_preds.copy())
        layer3.append(layer3_tree)
        layer3_sub_inds.append(subset_indices2)
        layer3_tree_subsets.append(df_subset2)

    else:
        Layer3_tree = Trees.DecisionTree()
        subset_indices2 = np.random.choice(len(layer2_train_preds), size=subset_size, replace=True)
        df_subset2 = layer2_train_preds[subset_indices2]
        Layer3_tree.train(df_subset2, subset_indices2)

        layer3_train_temp = Layer3_tree.sum_predictions(layer2_train_preds)
        layer3_data.append(layer3_train_temp.copy())

        layer3.append(Layer3_tree)
        layer3_tree_subsets.append(df_subset2)
        layer3_sub_inds.append(subset_indices2)


layer3_data = np.array(layer3_data)
layer3_data = modify_data.concat(layer3_data.transpose(), layer2_train_preds[:,-1])

MV_train = modify_data.Majority_voting(layer3_data)
#print("Major vote shape: ", MV_train[:,1].shape)
L3_train_accuracy = modify_data.score(MV_train[:,0], MV_train[:,1])
#MV_test = modify_data.Majority_voting(layer2_test_preds)
#L2_test_accuracy = layer2[0].score(MV_test[:,0], MV_test[:,1])

print("Layer 3 training accuraccy pre retrain: ", L3_train_accuracy)
#print("Layer 2 validation accuracy pre retraining: ", L2_test_accuracy)
accuracy = L3_train_accuracy
Data_sets = [train, new_data, layer2_train_preds, layer3_data]
Layers = [layer1, layer2, layer3]
subsets = [tree_indices, layer3_sub_inds]

#Data_sets = [train, new_data, layer2_train_preds]
#Layers = [layer1, layer2]
#subsets = [tree_indices, tree_subsets]

#test_result = modify_data.test_predict(Layers, test)
#print("Layer 3 test accuracy pre retrain: ", test_result)

#print(subsets[0])
depth = len(Layers)
print(depth)

import os

#optimaltrees
# Create folders for layer1 and layer2 trees
os.makedirs("testtrees/layer1training", exist_ok=True)
os.makedirs("testtrees/layer2training", exist_ok=True)
os.makedirs("testtrees/layer3training", exist_ok=True)

# Save all layer1 trees
for i, tree in enumerate(layer1):
    graph = tree.export_graph()
    filepath = f"testtrees/layer1training/tree_{i}"
    graph.render(filepath, format="png", cleanup=True)

# Save all layer2 trees
for i, tree in enumerate(layer2):
    graph = tree.export_graph()
    filepath = f"testtrees/layer2training/tree_{i}"
    graph.render(filepath, format="png", cleanup=True)

# Save all layer3 trees
for i, tree in enumerate(layer3):
    graph = tree.export_graph()
    filepath = f"testtrees/layer3training/tree_{i}"
    graph.render(filepath, format="png", cleanup=True)
#6, 25, 69, 93
print(layer1)
for i in range(20):
    np.random.seed(SEED + i)
    random.seed(SEED + i)
    print("Iteration", i+1, "Starts")

    # === Shapley ===
    print("Start shap single")
    ret_layerS_shap, ret_dataS_shap, best_testS_shap, best_featS_shap, best_subS_shap, best_ind_subS_shap = (
        backpropagationShapley.find_best_improvement_single(
            Layers[-2:], Data_sets[-3], Data_sets[-2], L3_train_accuracy,
            random_trees_included, distribution, Shapley_chosen_features, Shapley_number_tries
        )
    )

    print("Start shap propagate")
    ret_layerP_shap, ret_dataP_shap, best_testP_shap, best_subsP_shap, best_featsP_shap = (
        backpropagationShapley.find_best_improvement_propagate(
            Layers, Data_sets[:-1], L3_train_accuracy, depth,
            subsets, [], [], [],
            random_trees_included, distribution,
            Shapley_chosen_features, [], Shapley_number_tries
        )
    )

    print("shap single feature: ", best_featS_shap)
    print("shap propagate feature: ", best_featsP_shap)
    best_acc = max(best_testS_shap, best_testP_shap)

    if (best_testP_shap > best_testS_shap) & (accuracy - best_testP_shap != accuracy):
        print("Propagate > Singular")
        Layers = deepcopy(ret_layerP_shap)
        Data_sets[1:] = deepcopy(ret_dataP_shap)
        subsets = deepcopy(best_subsP_shap)
        accuracy = best_testP_shap
        print("shap Propagate accuracy", accuracy)

    elif (best_testS_shap > best_testP_shap) & (accuracy - best_testS_shap != accuracy):
        print("Propagate < Singular")
        Layers[-2:] = deepcopy(ret_layerS_shap)
        Data_sets[-2:] = deepcopy(ret_dataS_shap)
        for j in range(len(best_featS_shap)):
            subsets[-2][best_featS_shap[j]] = best_subS_shap[j].copy()
        subsets[-1] = deepcopy(best_ind_subS_shap)
        accuracy = best_testS_shap
        print("shap Single accuracy", accuracy)

    elif (best_testS_shap == best_testP_shap) & (best_testS_shap >= accuracy):
        print("Propagate == Singular")
        single_layers = deepcopy(Layers)
        single_layers[-2:] = deepcopy(ret_layerS_shap)
        propagate_layers = deepcopy(ret_layerP_shap)
        single_test_accuracy = modify_data.test_predict(single_layers, test)
        propagate_test_accuracy = modify_data.test_predict(propagate_layers, test)
        randomDouble = np.random.rand()
        if single_test_accuracy > propagate_test_accuracy or ((single_test_accuracy == propagate_test_accuracy) and randomDouble <= 0.5):
            print("Singular was better")
            Layers[-2:] = deepcopy(ret_layerS_shap)
            Data_sets[-2:] = deepcopy(ret_dataS_shap)
            for j in range(len(best_featS_shap)):
                subsets[-2][best_featS_shap[j]] = best_subS_shap[j].copy()
            subsets[-1] = deepcopy(best_ind_subS_shap)
            accuracy = best_testS_shap
        else:
            print("Propagate was better")
            Layers = deepcopy(ret_layerP_shap)
            Data_sets[1:] = deepcopy(ret_dataP_shap)
            subsets = deepcopy(best_subsP_shap)
            accuracy = best_testP_shap

    else:
        print("Original best")
        test_accuracy = modify_data.test_predict(Layers, test)
        print("if-statement in main accuracy post retrain (test data): ", test_accuracy)
        print("if-statement in main accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)
        if Shapley_chosen_features < max_feats:
            Shapley_chosen_features = int(Shapley_chosen_features + increase)
        else:
            Shapley_chosen_features = start_features
        continue

    test_accuracy = modify_data.test_predict(Layers, test)
    print("accuracy post retrain (test data): ", test_accuracy)
    print("accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)

    iteration_folder = f"testtrees/iteration_{i + 1}"
    os.makedirs(f"{iteration_folder}/layer1", exist_ok=True)
    os.makedirs(f"{iteration_folder}/layer2", exist_ok=True)
    os.makedirs(f"{iteration_folder}/layer3", exist_ok=True)

    for j, tree in enumerate(Layers[0]):
        graph = tree.export_graph()
        filepath = f"{iteration_folder}/layer1/tree_{j}"
        graph.render(filepath, format="png", cleanup=True)

    for j, tree in enumerate(Layers[1]):
        graph = tree.export_graph()
        filepath = f"{iteration_folder}/layer2/tree_{j}"
        graph.render(filepath, format="png", cleanup=True)

    for j, tree in enumerate(Layers[2]):
        graph = tree.export_graph()
        filepath = f"{iteration_folder}/layer3/tree_{j}"
        graph.render(filepath, format="png", cleanup=True)

end = datetime.now()
print("End Time: ", end)
print(f"Duration: {end - start}")
print("Final done")