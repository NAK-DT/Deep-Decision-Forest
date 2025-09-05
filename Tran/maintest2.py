import numpy as np
import pandas as pd
import Trees
import modify_data
import backpropagationShapley
import backpropagationExhaustive
from copy import deepcopy
from sklearn.datasets import load_digits, fetch_openml
import os
import time
from datetime import datetime
from sklearn.datasets import load_breast_cancer
#data = pd.read_csv('Datasets/mushroomfinal.csv')

start = datetime.now()
print("Start_time: ", start)

#data = pd.read_csv('C:\\Users\\David\\PycharmProjects\\Examensarbete\\Datasets\\SynthDatasets\\syntheticredundancy12.csv')


#data = pd.read_csv('Datasets/SynthDatasets/test.csv')
#data = np.array(data)
#np.random.shuffle(data)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
data = np.hstack((X.values, y.values.reshape(-1, 1)))
np.random.shuffle(data)

train, test = modify_data.train_test_split(data, 0.7)
'''
digits = load_digits()
X = digits.data
y = digits.target

data = np.hstack((X, y.reshape(-1, 1)))
np.random.shuffle(data)
train, test = modify_data.train_test_split(data, 0.7)
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
print("Class distribution:", dict(zip(unique, counts)))
layer1 = []
L1_data_holder = []
L1_preds = []
data_indeces = []
layer1_acc = []
all_layers = []


random_trees_included = False

NoT = 11
random_trees = 33
Shapley_chosen_features = 3
Shapley_number_tries = 1
increase = 3


distribution = NoT-random_trees-1
#Code for testing re-training function for a single tree in layer 2
for i in range(NoT):
    rand = Trees.DecisionTree()
    rand.random_train(train)
    layer1.append(rand)
    preds = rand.sum_predictions(test)
    L1_preds.append(preds)
    L1_data, data_index = modify_data.get_results(rand.base, [], [])
    L1_data_holder.append(L1_data)
    data_indeces.append(data_index)
data_indeces = np.array(data_indeces)
L1_data_holder = np.array(L1_data_holder)
print(L1_data_holder.shape, "   ", data_indeces.shape)
L1_data_holder, template_indeces, data_indeces = modify_data.rearange_data(L1_data_holder, data_indeces)
print(L1_data_holder.shape,"   ", data_indeces.shape)
#print("prediction shapes: ", np.shape(temp_preds))
#print("training shapes: ", np.shape(temp))
#print("Mean accuracy layer 1 (pre retrain): ", layer1_acc.mean())
#print(layer1[0].base.left.left.parent.parent.information_gain == layer1[0].base.information_gain == layer1[0].base.right.parent.information_gain)
#print(layer1[0].base.parent)
L1_preds = np.array(L1_preds)
L1_data_holder = np.array(L1_data_holder)
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

accuracy = L3_train_accuracy

#print("Layer 2 validation accuracy pre retraining: ", L2_test_accuracy)

Data_sets = [train, new_data, layer2_train_preds, layer3_data]
Layers = [layer1, layer2, layer3]
subsets = [tree_indices, layer3_sub_inds]

test_result = modify_data.test_predict(Layers, test)
print("Layer 3 test accuracy pre retrain: ", test_result)
test_accuracy = test_result

#print(subsets[0])
depth = len(Layers)
print(depth)
#print("train: ", len(train[:,0]))

for i in range(10):
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

    print("Start exhaustive single")
    # === Exhaustive ===
    ret_layerS_exh, ret_dataS_exh, best_testS_exh, best_featS_exh, best_indS_exh, best_layerS_exh = (
        backpropagationExhaustive.find_best_improvement_single(
            Layers[-2:], Data_sets[-3], Data_sets[-2], L3_train_accuracy, subsets[-2], distribution, random_trees_included
        )
    )

    print("Start exhaustive propagate")
    ret_layerP_exh, ret_dataP_exh, best_testP_exh, best_subsP_exh, best_featsP_exh = (
        backpropagationExhaustive.find_best_improvement_propagate(
            Layers, Data_sets[:-1], L3_train_accuracy, depth, subsets, [], [], [],
            include_random_trees=random_trees_included,
            distribution=distribution
        )
    )

    print("Shapley single feature: ", best_featS_shap)
    print("Shapley propagate feature: ", best_featsP_shap)
    print("Exhaustive single feature: ", best_featS_exh)
    print("Exhaustive propagate feature: ", best_featsP_exh)
    # === Choose best ===
    #best_acc = max(best_testS_shap, best_testP_shap, best_testS_exh, best_testP_exh)
    #best_acc = max(best_testS_shap, best_testP_shap, best_testS_exh, best_testP_exh)

    #test 1
    #best_acc = max(best_testS_exh, best_testP_exh)
    #test 2
    best_acc = max(best_testS_shap, best_testP_shap)

    if (best_testP_shap > best_testS_shap) & (accuracy - best_testP_shap != accuracy):
        print("Propagate > Singular")
        Layers = deepcopy(ret_layerP_shap)
        Data_sets[1:] = deepcopy(ret_dataP_shap)
        subsets = deepcopy(best_subsP_shap)
        accuracy = best_testP_shap
        print("Shapley Propagate accuracy", accuracy)

    elif (best_testS_shap > best_testP_shap) & (accuracy - best_testS_shap != accuracy):
        print("Propagate < Singular")
        Layers[-2:] = deepcopy(ret_layerS_shap)
        Data_sets[-2:] = deepcopy(ret_dataS_shap)
        for j in range(len(best_featS_shap)):
            subsets[-2][best_featS_shap[j]] = best_subS_shap[j].copy()
        subsets[-1] = deepcopy(best_ind_subS_shap)
        accuracy = best_testS_shap
        print("Shapley Single accuracy", accuracy)
    else:
        print("Original best")
        print("if-statement in main accuracy post retrain (test data): ", test_accuracy)
        print("if-statement in main accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)

        continue

    test_accuracy = modify_data.test_predict(Layers, test)
    print("accuracy post retrain (test data): ", test_accuracy)
    print("accuracy post retrain (train data): ", accuracy, " for iteration: ", i + 1)


end = datetime.now()
print("End Time: ", end)
print(f"Duration: {end - start}")
print("Final done")












    #if best_acc == best_testP_exh:
    #    print("[✓] Using Exhaustive Propagate")
    #    Layers = deepcopy(ret_layerP_exh)
    #    Data_sets[1:] = deepcopy(ret_dataP_exh)
    #    subsets = deepcopy(best_subsP_exh)

    #elif best_acc == best_testS_exh:
    #    print("[✓] Using Exhaustive Single")
    #    Layers[-2:] = deepcopy(ret_layerS_exh)
    #    Data_sets[-2:] = deepcopy(ret_dataS_exh)
        # If you eventually modify exhaustive to return subset updates, handle them here
    #else:
    #    print("No improvement. Adjusting feature count.")
        #if Shapley_chosen_features < 30:
        #    Shapley_chosen_features += increase
    # === Evaluate after update ===


'''
import numpy as np
import pandas as pd
import Trees
import modify_data
#import backpropagation
from copy import deepcopy
from sklearn.datasets import load_digits
#data = pd.read_csv('Datasets/mushroomfinal.csv')
digits = load_digits()
X = digits.data
y = digits.target

#data = pd.read_csv('/Users/hehexdddd/PycharmProjects/Examensarbete/Datasets/SynthDatasets/syntheticredundancy12.csv')
#data = pd.read_csv('Datasets/SynthDatasets/test.csv')
#data = np.array(data)
#np.random.shuffle(data)
#train, test = modify_data.train_test_split(data, 0.7)

data = np.hstack((X, y.reshape(-1, 1)))
np.random.shuffle(data)
train, test = modify_data.train_test_split(data, 0.7)
#Code for mutliple trees
unique, counts = np.unique(data[:, -1], return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))
layer1 = []
L1_data_holder = []
L1_preds = []
data_indeces = []
layer1_acc = []
all_layers = []


random_trees_included = False

NoT = 31
random_trees = 33
Shapley_chosen_features = 3
Shapley_number_tries = 1
increase = 3



distribution = NoT-random_trees-1
#Code for testing re-training function for a single tree in layer 2
for i in range(NoT):
    rand = Trees.DecisionTree()
    rand.random_train(train)
    layer1.append(rand)
    preds = rand.sum_predictions(test)
    L1_preds.append(preds)
    L1_data, data_index = modify_data.get_results(rand.base, [], [])
    L1_data_holder.append(L1_data)
    data_indeces.append(data_index)
data_indeces = np.array(data_indeces)
L1_data_holder = np.array(L1_data_holder)
print(L1_data_holder.shape, "   ", data_indeces.shape)
L1_data_holder, template_indeces, data_indeces = modify_data.rearange_data(L1_data_holder, data_indeces)
print(L1_data_holder.shape,"   ", data_indeces.shape)
#print("prediction shapes: ", np.shape(temp_preds))
#print("training shapes: ", np.shape(temp))
#print("Mean accuracy layer 1 (pre retrain): ", layer1_acc.mean())
#print(layer1[0].base.left.left.parent.parent.information_gain == layer1[0].base.information_gain == layer1[0].base.right.parent.information_gain)
#print(layer1[0].base.parent)
L1_preds = np.array(L1_preds)
L1_data_holder = np.array(L1_data_holder)
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

Data_sets = [train, new_data, layer2_train_preds, layer3_data]
Layers = [layer1, layer2, layer3]
subsets = [tree_indices, layer3_sub_inds]

test_result = modify_data.test_predict(Layers, test)
print("Layer 3 test accuracy pre retrain: ", test_result)

print(subsets[0])
depth = len(Layers)
print(depth)

#print("train: ", len(train[:,0]))

for i in range(50):
    print("Iteration ", i+1, " Starts")
    ret_layerS, ret_dataS, best_testS, best_featS,  best_subS, best_ind_subS = backpropagation.find_best_improvement_single(Layers[-2:], Data_sets[-3], Data_sets[-2], L3_train_accuracy, random_trees_included, distribution, Shapley_chosen_features, Shapley_number_tries)
    ret_layerP, ret_dataP, best_testP, best_subsP = backpropagation.find_best_improvement_propagate(Layers, Data_sets[:-1], L3_train_accuracy, depth, subsets, [], [], [], random_trees_included, distribution, Shapley_chosen_features, [], Shapley_number_tries)

    if (best_testP > best_testS) & (L3_train_accuracy - best_testP != L3_train_accuracy):
        print("Propagate > Singular")
        Layers = deepcopy(ret_layerP)
        Data_sets[1:] = deepcopy(ret_dataP)
        subsets = deepcopy(best_subsP)

    elif (best_testS > best_testP) & (L3_train_accuracy - best_testS != L3_train_accuracy):
        print("Propagate < Singular")
        Layers[-2:] = deepcopy(ret_layerS)
        Data_sets[-2:] = deepcopy(ret_dataS)
        for j in range(len(best_featS)):
            subsets[-2][best_featS[j]] = best_subS[j].copy()
        subsets[-1] = deepcopy(best_ind_subS)
    else:
        print("Original best")
        if Shapley_chosen_features < 11:
            Shapley_chosen_features = Shapley_chosen_features + increase

    checker = []
    for tree in Layers[-1]:
        train_res = tree.sum_predictions(Data_sets[2])
        if not len(checker):
            checker = train_res
        else:
            checker = modify_data.concat(checker, train_res)

    checker = modify_data.concat(checker, Data_sets[2][:,-1])

    MV_training = modify_data.Majority_voting(checker)
    accuracy = modify_data.score(MV_training[:,0], MV_training[:,1])
    test_accuracy = modify_data.test_predict(Layers, test)
    print("accuracy post retrain (test data): ", test_accuracy)
    print("accuracy post retrain (train data): ", accuracy, " for iteration: ", i+1)
    L3_train_accuracy = accuracy
print("Hello")
'''
