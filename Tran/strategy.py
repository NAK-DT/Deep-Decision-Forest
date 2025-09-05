from collections import Counter
from random import random
import backpropagationShapley
import Trees
import Node
import modify_data
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Tuple, Optional


def entropy(y):  # Calculate the entropy of the datapoints in the node
    y = np.array(y)
    labels = np.unique(y)
    entropy = 0
    for label in labels:
        number = y[y == label]  # Checks the number of datapoints that corresponds to the label
        amount = len(number) / len(y)  # devides by the number of datapoints
        entropy -= amount * np.log2(amount)  # Adds to the entropy
    return entropy

def conditional_entropy(x, y):
    total = len(y)
    y_counter = Counter(y)
    xy = list(zip(x, y))
    condEntropy = 0.0
    for y_val, y_count in y_counter.items():
        x_given_y = [x_val for x_val, y_val2 in xy if y_val2 == y_val]
        weight = y_count / total
        condEntropy += weight * entropy(x_given_y)
    return condEntropy

def symmetrical_uncertainty(Data, feature):
    X, y = Data[:, :-1], Data[:, -1]  # split ground truth from training parameters
    su = 2.0 * ((entropy(X[:, feature]) + entropy(y) + conditional_entropy(X[:, feature], y))/ (entropy(y) + entropy(X[:, feature])))
    return su

def su_computation(split_nodes, numFeats, bias_indices = None, bias_classes = None):
    if bias_indices:
        temp_indices_holder = {}
        temp_class_holder = {}

        # Loop through each list of indices and corresponding classes in bias_indices and bias_classes
        for i in range(len(bias_indices)):
            for j in range(len(bias_indices[i])):
                index = bias_indices[i][j]
                class_label = bias_classes[i][j]

                # If the index is already in the temp_indices_holder, append the new class
                if index in temp_indices_holder:
                    temp_class_holder[index].append(class_label)
                else:
                    # If it's a new index, add it to temp_indices_holder and start a new class list
                    temp_indices_holder[index] = True  # This can be any value; we just care about existence
                    temp_class_holder[index] = [class_label]  # Initialize with the first class label

        # Convert the dictionaries back into lists (if needed)
        final_indices = list(temp_indices_holder.keys())
        final_classes = [temp_class_holder[index] for index in final_indices]

        summed_indices, summed_classes = backpropagationShapley.sum_classes(final_indices, final_classes)
    else:
        summed_indices = None
        summed_classes = None


    su_scoreList = [None] * numFeats
    totalDepth_list = []
    avgDepth_list = []
    for collection in split_nodes:
        node = collection[1]
        data = node.data
        feature = node.feature
        if feature is None:
            print(" Fault: ", feature, node, collection, node.value, data)
            continue
        su = symmetrical_uncertainty(data, feature)
        depth = node.depth
        if su_scoreList[feature] is None:
            su_scoreList[feature] = [su, depth, 1, feature]
        else:
            su_scoreList[feature][0] += su
            su_scoreList[feature][1] += depth
            su_scoreList[feature][2] += 1

        #print("Feature: ", feature, " Depth: ", depth)

    for item in su_scoreList:
        if item is None:
            continue
        totalDepth_list.append([item[3], item[0] / item[1]])
        avgDepth_list.append([item[3], item[0] / (item[1] / item[2])])

    #print("Total depth: ", totalDepth_list)
    #print("average depth: ", avgDepth_list)
    #print("Hello")


    return totalDepth_list, summed_indices, summed_classes, True




def Shapley_compute(Layer, Data, Bad_features, bias_indices = None, bias_classes = None):
    if bias_indices:
        temp_indices_holder = {}
        temp_class_holder = {}

        # Loop through each list of indices and corresponding classes in bias_indices and bias_classes
        for i in range(len(bias_indices)):
            for j in range(len(bias_indices[i])):
                index = bias_indices[i][j]
                class_label = bias_classes[i][j]

                # If the index is already in the temp_indices_holder, append the new class
                if index in temp_indices_holder:
                    temp_class_holder[index].append(class_label)
                else:
                    # If it's a new index, add it to temp_indices_holder and start a new class list
                    temp_indices_holder[index] = True  # This can be any value; we just care about existence
                    temp_class_holder[index] = [class_label]  # Initialize with the first class label

        # Convert the dictionaries back into lists (if needed)
        final_indices = list(temp_indices_holder.keys())
        final_classes = [temp_class_holder[index] for index in final_indices]

        summed_indices, summed_classes = backpropagationShapley.sum_classes(final_indices, final_classes)
        Data = deepcopy(Data[summed_indices])
        bias_classes = summed_classes.copy()
    else:
        summed_indices = None
        summed_classes = None

    samples, features = Data.shape
    feature_size = features-1




    if feature_size < 100:
        sub_features = int(np.sqrt(feature_size))
    else:
        sub_features = int(feature_size * 0.1)


    subset_size = int(0.6*len(Data))

    Feature_exist: List[Optional[Tuple[float, int]]] = [None] * feature_size
    Feature_not_exist: List[Optional[Tuple[float, int]]] = [None] * feature_size

    Data_copy = deepcopy(Data)

    '''shapley_trees = []
das
    for j in range(len(Layer)):
        temp_tree = Trees.DecisionTree()
        subset_index = np.random.choice(len(Data_copy), subset_size, replace=True)
        subset = Data[subset_index]

        temp_tree.train(subset, subset_index)
        shapley_trees.append(temp_tree)'''

    iterations = int(np.log(feature_size) * 50)
    n_classes = len(np.unique(Data[:, -1])) #amount of classes
    for i in range(iterations):


        swapped_features = random.sample(range(feature_size), sub_features)
        not_swapped_features = set(range(feature_size)) - set(swapped_features)
        random_features = backpropagationShapley(feature_size, samples, n_classes) #<-- added n_classes
        Data_copy[:,swapped_features] = random_features[:,swapped_features].copy()
        shapley_preds = []
        for tree in Layer:

            preds = tree.sum_predictions(Data_copy)

            shapley_preds.append(preds)

        Data_copy[:,swapped_features] = Data[:,swapped_features].copy()

        if bias_classes:
            shapley_preds.append(bias_classes.copy())
        else:
            shapley_preds.append(Data[:,-1].copy())
        shapley_preds = np.array(shapley_preds).transpose()
        MV = modify_data.Majority_voting(shapley_preds)

        temp_accuracy = modify_data.score(MV[:,0], MV[:,1])

        for feat in swapped_features:
            if Feature_exist[feat] is None:
                Feature_exist[feat] = (temp_accuracy, 1)
            else:
                Feature_exist[feat] = (Feature_exist[feat][0] + temp_accuracy, Feature_exist[feat][1] + 1)

        for feat in not_swapped_features:
            if Feature_not_exist[feat] is None:
                Feature_not_exist[feat] = (temp_accuracy, 1)
            else:
                Feature_not_exist[feat] = (Feature_not_exist[feat][0] + temp_accuracy, Feature_not_exist[feat][1] + 1)



    feature_exist_accuracies = []
    feature_not_exist_accuracies = []

    for i in range(len(Feature_not_exist)):
        if i < len(Feature_exist):
            if Feature_exist[i] is None:
                continue
            else:
                feature_exist_accuracies.append(Feature_exist[i][0]/Feature_exist[i][1])

        if Feature_not_exist[i] is None:
            continue
        else:
            feature_not_exist_accuracies.append(Feature_not_exist[i][0]/Feature_not_exist[i][1])


    '''smallest_difference = 100

    worst_feature = None

    for i in range(feature_size):

        if (abs(feature_exist_accuracies[i]-feature_not_exist_accuracies[i]) < smallest_difference) & ( i not in Bad_features):
            smallest_difference = abs(feature_exist_accuracies[i]-feature_not_exist_accuracies[i])
            worst_feature = i'''

    accuracy_differences = abs(np.array(feature_exist_accuracies)-np.array(feature_not_exist_accuracies))

    indexed_arr = [(i, v) for i, v in enumerate(accuracy_differences) if i not in Bad_features]



    return indexed_arr, summed_indices, summed_classes, True

def average_node_depth(features, layer_nodes, bias_indices = None, bias_classes = None):
    if bias_indices:
        temp_indices_holder = {}
        temp_class_holder = {}

        # Loop through each list of indices and corresponding classes in bias_indices and bias_classes
        for i in range(len(bias_indices)):
            for j in range(len(bias_indices[i])):
                index = bias_indices[i][j]
                class_label = bias_classes[i][j]

                # If the index is already in the temp_indices_holder, append the new class
                if index in temp_indices_holder:
                    temp_class_holder[index].append(class_label)
                else:
                    # If it's a new index, add it to temp_indices_holder and start a new class list
                    temp_indices_holder[index] = True  # This can be any value; we just care about existence
                    temp_class_holder[index] = [class_label]  # Initialize with the first class label

        # Convert the dictionaries back into lists (if needed)
        final_indices = list(temp_indices_holder.keys())
        final_classes = [temp_class_holder[index] for index in final_indices]

        summed_indices, summed_classes = backpropagationShapley.sum_classes(final_indices, final_classes)
    else:
        summed_indices = None
        summed_classes = None


    indexed_arr =[]


    for feat in features:
        temp_nodes = [node_info for node_info in layer_nodes if node_info[2] == feat]
        summed_depth = 0
        for node in temp_nodes:
            summed_depth = summed_depth + node[1].depth

        summed_depth = summed_depth/len(temp_nodes)

        indexed_arr.append((feat, summed_depth))

    '''for index, feature_nodes in enumerate(layer_nodes):
        summed_depth = 0
        for split_node in feature_nodes:
            summed_depth += split_node[1].depth

        summed_depth = summed_depth/len(feature_nodes)

        indexed_arr.append((features[index], summed_depth))'''

    return indexed_arr, summed_indices, summed_classes, False














