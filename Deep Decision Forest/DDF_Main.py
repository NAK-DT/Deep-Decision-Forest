import heapq

import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

import Trees
import modify_data
import backpropagation
from copy import deepcopy
from datetime import datetime

start = datetime.now()
print("Start_time: ", start)


""" 
Preprocessing start

Everything between here and the green end statement should be replaced with the rows from the datasetpreprocessing map files
for each individual datasets to replicate the dataset preprocessing from the paper (imports can be before) 
"""
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

data = np.hstack([X.values, y_encoded.reshape(-1,1)])
np.random.shuffle(data)


train_t, test = modify_data.train_test_split(data, 0.7)
train_t, val = modify_data.train_test_split(train_t, 0.8)


train = train_t.copy()


unique, counts = np.unique(data[:, -1], return_counts=True)
train_unique, train_count = np.unique(train[:, -1], return_counts=True)
test_unique, test_count = np.unique(test[:, -1], return_counts=True)

"""
Preprocessing end
"""


print("Class distribution:", dict(zip(unique, counts)))
print("Class distribution (train):", dict(zip(train_unique, train_count)))
print("Class distribution (test):", dict(zip(test_unique, test_count)))

"""
Here begins the model settings
"""

#Setup
Use_NoT3 = True #Decides if final layer should use fewer trees than the other layer (True mean fewer, False means equal)
NoT_Multiplier = 1.5 #The multiplier for number of trees in final layer (based on number of epochs)


use_weights = False #Decides if the model should use weighted majority voting


NoT = 60 #How many trees should be in all layers (excluding final if Use_NoT3 is set to True)
epochs = 10 #Number of epochs the model should run (also decides number of trees in final layer if Use_NoT3 is True)

#Backpropagation
Use_increase = True #Decides if the number of features that should be retrained increase or decreases after not finding an improvement
Shapley_chosen_features = 50 # The number of features to be retrained at the start

increase = 0 #How much the retraining features should increase if Use_increase is set to True
max_feats = 30 #Maximum allowed features to retrain (If larger reset to initial number of features)

divided = 2 #How much the number of retrained feature is divided by when Use_increase is set to False
min_feats = 2 #The minimum allowed features to retrain before reset.

start_features = Shapley_chosen_features #Reset save for when model reaches outside the number of feature retrain threshold


Shapley_number_tries = 1 #Allows the model to do multiple tests if an improvement could not be found during an epoch

Use_AD = True #Decides if the model uses Average Depth, or Average SU Score during back propagation
number_threshs = 3 #Decides how many thresholds zones the model is allowed to test to find the best area for each sample


#Retraining
M_CANDIDATES = 50 #Decides how many trees the model is allowed to create per retraining slot in the final layer to attempt to find an improvement
BAG_FRAC = 0.63 #Decides the size of the training dataset for the final layer tree retraining
Bootstrap = True #Decides if duplicate samples is allowed to be used in final layer retraining
use_retrain_weights = False #Decides if the model should use weights when retraining final layer trees
K = 5 #Decides the number of trees to be retrained in the final layer


COOLDOWN_LEN = 3 #Decides the number of epochs a set of features is put into cooldown if they do not produce an improvement

"""
End of model settings section
"""

if K > epochs * NoT_Multiplier:
    raise ValueError("K is greater than epochs * NoT_Multiplier. You can not retrain more trees than you have.")

if Shapley_chosen_features > NoT or max_feats > NoT:
    raise ValueError("Number of trees smaller than potential trees that can being retrained. You can not retrain more trees than you have.")

if not Use_increase and divided == 0:
    raise ValueError("Divided is set to 0. Will cause mathematical error if continued.")



layer1 = []
L1_data_holder = []
L1_preds = []
layer1_acc = []
all_layers = []

for i in range(NoT): #create first layer
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
L1_preds = np.array(L1_preds)
L1_data_holder = np.array(L1_data_holder.transpose())
new_data = modify_data.concat(L1_data_holder, train[:,-1])
#print("Data after changes: ", new_data)
new_preds = modify_data.concat(L1_preds.transpose(), test[:,-1])


print("total Data: ", np.sum(new_data[:,-1]==True))
layer2 = []
layer2_train_preds = []
layer2_test_preds = []
subset_size = int(0.6 * len(new_data))

tree_subsets = []
tree_indices = []
for i in range(NoT): #create second layer
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
for i in range(NoT3): #create final layer
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
L3_train_accuracy = modify_data.score(MV_train[:,0], MV_train[:,1])

print("Layer 3 training accuraccy pre retrain: ", L3_train_accuracy)

accuracy = L3_train_accuracy

Data_sets = [train, new_data, layer2_train_preds, layer3_data]
Layers = [layer1, layer2, layer3]

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



iter_start = datetime.now()
for i in range(epochs):

    print(f"\n⏱️ Iteration {i + 1} starts at {iter_start.strftime('%H:%M:%S')}...")
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
        backpropagation.find_best_improvement_single(
            Layers[-2:], Data_sets[-3], Data_sets[-2:], L3_train_accuracy, Shapley_chosen_features, Shapley_number_tries, worst_tree_idx, Use_AD, number_threshs, deepcopy(layer3_sub_inds), BAG_FRAC, M_CANDIDATES, use_retrain_weights, Bootstrap
        )
    )

    print("Start shap propagate")
    ret_layerP_shap, ret_dataP_shap, best_testP_shap, best_featsP_shap, accepted_count_P, accepted_slots_P, propagate_L3_inds, propagate_weights = (
        backpropagation.find_best_improvement_propagate(
            Layers, Data_sets, L3_train_accuracy, depth, [], [], [],
            Shapley_chosen_features, [], Shapley_number_tries, [worst_tree_idx], Use_AD, number_threshs, deepcopy(layer3_sub_inds), BAG_FRAC, M_CANDIDATES, use_retrain_weights, Bootstrap
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

                if Use_increase:

                    if Shapley_chosen_features < max_feats:
                        Shapley_chosen_features = int(Shapley_chosen_features + increase)
                    else:
                        Shapley_chosen_features = start_features
                    continue

                else:
                    if Shapley_chosen_features > min_feats:
                        Shapley_chosen_features = int(Shapley_chosen_features/divided)
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

                if Use_increase:

                    if Shapley_chosen_features < max_feats:
                        Shapley_chosen_features = int(Shapley_chosen_features + increase)
                    else:
                        Shapley_chosen_features = start_features
                    continue

                else:
                    if Shapley_chosen_features > min_feats:
                        Shapley_chosen_features = int(Shapley_chosen_features / divided)
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

            if Use_increase:

                if Shapley_chosen_features < max_feats:
                    Shapley_chosen_features = int(Shapley_chosen_features + increase)
                else:
                    Shapley_chosen_features = start_features
                continue

            else:
                if Shapley_chosen_features > min_feats:
                    Shapley_chosen_features = int(Shapley_chosen_features / divided)
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

        if Use_increase:

            if Shapley_chosen_features < max_feats:
                Shapley_chosen_features = int(Shapley_chosen_features + increase)
            else:
                Shapley_chosen_features = start_features
            continue

        else:
            if Shapley_chosen_features > min_feats:
                Shapley_chosen_features = int(Shapley_chosen_features / divided)
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
