import math

import Node
import random
from graphviz import Digraph
import numpy as np
import random

def generate_threshold(y, num_bins=10):
    """
    Generate a threshold:
    - If binary classification: return float thresholds with decimal precision.
    - If multiclass classification: return integer thresholds.

    Parameters:
        y (np.ndarray): Ground truth label vector (1D).
        num_bins (int): Number of bins/steps (used only for binary).

    Returns:
        int or float: A randomly selected threshold.
    """
    unique_vals = np.unique(y)

    '''if len(unique_vals) == 2:
        # Binary classification → use float thresholds
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if y_min == y_max:
            return y_min  # Only one value
        step = (y_max - y_min) / num_bins
        bins = [round(y_min + i * step, 6) for i in range(num_bins + 1)]
        return random.choice(bins)
    else:'''
    # Multiclass → use integer thresholds from range of y
    if np.max(y) - np.min(y) <= 1:
        return random.uniform(np.min(y), np.max(y))
    y_min, y_max = int(np.min(y)), int(np.max(y))
    return random.randint(y_min, y_max)


class DecisionTree: #Used for creating trees of random, biased and greedy variations
    def __init__(self, Maxdepth = 10, min_gain = 5e-3, min_samples = 10):
        self.Maxdepth = Maxdepth #Sets the maximum number of layers fo a single tree
        self.min_gain = min_gain #Minimum gain value for allowing to split a node
        self.min_samples = min_samples #Minimum samples to allow a node split

    #print decision trees via graphs
    def export_graph(self, node=None, graph=None):
        if node is None:
            node = self.base # iniatilize the starting node as the base node of the decision tree if there is no node.
            graph = Digraph() #calls the visualization library graphviz.

        if node.value is not None: #checks if current node is a leaf node, leaf node has value and therefore does not split any further.
            graph.node(str(id(node)), f"Leaf: {node.value}\nAcc: {node.accuracy:.2f}") #adds the current node as a LEAF IN THE GRAPH, stores its prediction and accuracy.
        else:
            threshold = f"{node.threshold:.2f}" if node.threshold is not None else "N/A"
            gain = f"{node.information_gain:.2f}" if node.information_gain is not None else "N/A"
            graph.node(str(id(node)), f"Feature {node.feature}\nThreshold: {threshold}\nGain: {gain}")
            #if current node is not a leaf, it adds node as non-feature leaf, stores threshold value, information gain and feature index to split at this node.
            if node.left: #checks if node has a left child
                graph.edge(str(id(node)), str(id(node.left)), "Left") #adds an edge from the current node to the left child in the graph
                self.export_graph(node.left, graph) #is added to the subtree through recursive calling
            if node.right:
                graph.edge(str(id(node)), str(id(node.right)), "Right") #same thing here for left child
                self.export_graph(node.right, graph)
        return graph

    def split(self, data, feature, threshold, data_indices, alt_feature=None):
        data = np.asarray(data)
        data_indices = np.asarray(data_indices)

        # Boolean mask: True → left branch, False → right branch
        mask = data[:, feature] <= threshold  # shape (N,)

        # Slice data and indices
        left_data = data[mask]
        right_data = data[~mask]
        left_index = data_indices[mask]
        right_index = data_indices[~mask]

        # Handle alternate feature values
        if alt_feature is not None and len(alt_feature) > 0:
            alt_feature = np.asarray(alt_feature)

            if alt_feature.shape[0] != data.shape[0]:
                raise ValueError(
                    f"[split] Mismatch: alt_feature has {alt_feature.shape[0]} rows, "
                    f"but data has {data.shape[0]}. You must subset alt_feature using data_indices before calling split()."
                )

            left_feature = alt_feature[mask]
            right_feature = alt_feature[~mask]
        else:
            # Maintain output structure even if alt_feature not provided
            left_feature = []
            right_feature = []

        return left_data, right_data, left_index, right_index, left_feature, right_feature

    def entropy(self, y): #Calculate the entropy of the datapoints in the node
        labels = np.unique(y)
        entropy = 0
        for label in labels:
            number = y[y == label] #Checks the number of datapoints that corresponds to the label
            amount = len(number)/len(y) #devides by the number of datapoints
            entropy -= amount * np.log2(amount) #Adds to the entropy

        return entropy

    def information_gain(self, OG, D1, D2):
        OGE = self.entropy(OG) #Calculate entropy for entire dataset in a node
        D1E = self.entropy(D1) #Calculate entropy for data being sent to the left child node
        D2E = self.entropy(D2) #Calculate entropy for data being sent to the right child node
        D1W = len(D1)/len(OG) #calculate the weight of the left node
        D2W = len(D2)/len(OG) #calculate teh weight of the right node
        IG = OGE - (D1W*D1E + D2E*D2W) #calculagte the information gain

        return IG


    def find_best_Node_split(self, data, features, data_indices, biased_value, alt_feature = None): #Find the best feature and threshold combination for the split of the current node

        best_split = {"gain": -1, "feature": None, "threshold": None, "LeftData": None, "RightData": None, "LeftIndex": None, "RightIndex": None, "Left_feat": None, "Right_feat": None} #Initialise best_split
        for feature in features:


            #if biased_value:
                #print("Set y as thresh")
            #    feature_vals = alt_feature
            #    treshold = np.unique(feature_vals)
            #else:
            feature_vals = data[:, feature]  # For every feature extract every variable from said features
            treshold = np.unique(feature_vals) #Set threshold to every unique variable in feature
            #print("threshold: ", treshold)
            for tresh in treshold:

                split_left, split_right, left_data_index, right_data_index, left_feat, right_feat  = self.split(data, feature, tresh, data_indices, alt_feature) #For every threshold split the data based on feature and threshold
                #print("Lengths: ", len(split1), "  ", len(split2))
                if len(split_left)>0 and len(split_right)>0:# If both splits have data then the information gain is calculated

                    if alt_feature is not None and len(alt_feature) > 0:
                        y, y_left, y_right = np.array(alt_feature), np.array(left_feat), np.array(right_feat)
                    else:
                        y, y_left, y_right = data[:,-1], split_left[:,-1], split_right[:,-1]
                    gain = self.information_gain(y, y_left, y_right)
                    #print("gain: ", gain)
                    if gain > best_split["gain"]: # If the gain from information gain is better than the current best split gain then a new best split is added
                        best_split["gain"] = gain
                        best_split["threshold"] = tresh
                        #if alt_feature is not None:
                        #    best_split["feature"] = -1
                        #else:
                        best_split["feature"] = feature
                        best_split["LeftData"] = split_left
                        best_split["RightData"] = split_right
                        best_split["LeftIndex"] = left_data_index
                        best_split["RightIndex"] = right_data_index
                        if alt_feature is not None:
                            best_split["Left_feat"] = left_feat
                            best_split["Right_feat"] = right_feat

        return best_split #Return the best split once every combination has been tested.

    def leaf_value(self, y): # chooses the value for a leaf node based on the most common class present.
        y = list(y)
        return max(y, key=y.count)

    def yggdrasil(self, Data, depth, biased_value, use_subsampling, worst_index = None, data_indices = None, alt_features = None): #Implementation for greedy decision tree
        #print("Worst index contains: ", worst_index)

        if worst_index is not None:
            #print("Biased training")
            #print("Data shape: ", Data.shape)
            Data = Data[worst_index]
            data_indices = worst_index

        X, y = Data[:, :-1], Data[:, -1] #split ground truth from training parameters
        #print(X.shape, " ", y.shape)
        samples, features = X.shape #acquire the number of datapoints, and features present in the data


        if use_subsampling:
            sub_features = random.sample(range(features), max(2, int(math.sqrt(features))))
        else:
            sub_features = random.sample(range(features), max(2, int(features)))

        #if alt_features is not None:
        #    features = 1

        if data_indices is None:
            data_indices = np.arange(samples) #If no previous indices are recorded make a simple array in range 0 to length of data

        if samples >= self.min_samples and depth < self.Maxdepth: #if the tree is smaller than max depth or have more datapoints than minimum allow the node to attempt a split
            #print("I have reached this")
            best_split = self.find_best_Node_split(Data, sub_features, data_indices, biased_value, alt_features) #Finds the best split
            #print(best_split["gain"])
            if best_split["gain"] > self.min_gain: #if the gain is larger than minimum allowed the nod will be split
                #print("next layer")
                #print("Shape of left data: ", best_split["LeftData"].shape)
                #print("Shape of right data: ", best_split["RightData"].shape)
                LN = self.yggdrasil(best_split["LeftData"], depth + 1, biased_value, use_subsampling, worst_index = None, data_indices=best_split["LeftIndex"], alt_features= best_split["Left_feat"]) #create a left node using allocated data and indices. Adds +1 to current depth
                RN = self.yggdrasil(best_split["RightData"], depth + 1, biased_value, use_subsampling, worst_index = None,  data_indices=best_split["RightIndex"], alt_features= best_split["Right_feat"])#does the same for a right node

                if (alt_features is not None) & (depth == 1):
                    parent = Node.Node(Data, best_split["feature"], best_split['threshold'], best_split['gain'], LN, RN, depth=depth, alt_feat = alt_features, data_index= data_indices)
                else:
                    parent = Node.Node(Data, best_split["feature"], best_split['threshold'], best_split['gain'], LN, RN, depth=depth, data_index= data_indices) #create the parent node for the two nodes

                LN.parent = RN.parent = parent #assign parent

                return parent

        correct = 0
        #print("Data in the node: ", Data.shape)
        if alt_features is not None and len(alt_features) > 0:
            leaf_value = self.leaf_value(alt_features)
            for sample in alt_features:
                if sample == leaf_value:
                    correct += 1
            accuracy = correct / len(alt_features)
        else:
            leaf_value = self.leaf_value(y)
            for point in y:
                if point == leaf_value:
                    correct += 1
            accuracy = correct / len(y)
        #print("accuracy", accuracy, "\n\n")
        #print("y: ", len(Data))
        #print("leaf_value: ", leaf_value)
        #print("Node acc trained", accuracy)
        #print("Data: ", Data.shape)
        #print("Number: ", np.sum(y == leaf_value))
        return Node.Node(data=Data, value=leaf_value, accuracy = accuracy, data_index=data_indices, depth=depth, NoDP=len(Data)) #return leaf node

    def random_tree(self, Data, depth, index = None, used_features = None, designated_class = None): #implementation for a random tree
        X, y = Data[:, :-1], Data[:, -1] #initial code is the same as yggdrasil
        samples, features = X.shape

        if used_features is None:
            used_features = []
        if index is None:
            index = np.arange(samples)

        if samples >= self.min_samples and depth < self.Maxdepth:

            feature_idx = random.randint(0, features - 1) #feature is chosen at random
            while (np.unique(X[:, feature_idx]).size == 1) & (depth == 1):
                #while feature_idx in used_features:
                #print(np.unique(X[:, feature_idx]))
                feature_idx = random.randint(0, features - 1)

            #print(np.unique(X[:, feature_idx]))
            #print(f"Selected feature: {feature_idx}, Unique values: {np.unique(X[:, feature_idx])}")
            feature_vals = X[:, feature_idx]
            used_features = used_features.copy()
            used_features.append(feature_idx)

            if feature_vals.dtype == bool:
                    # Handle boolean features
                feature_vals = feature_vals.astype(int)  # Convert to integers

            if len(np.unique(feature_vals)) > 1: #if number of variables is greater than 1
                feature_min, feature_max = feature_vals.min(), feature_vals.max() #find the largest and smallest variables in the random feature
                #percentage = random.uniform(0.1, 0.9) #get a random percentage
                threshold = generate_threshold(feature_vals, num_bins=10)
                #print(threshold)
                while (depth == 1 ) and ((threshold <= np.min(feature_vals)) or (threshold > np.max(feature_vals))):
                    threshold = generate_threshold(feature_vals, num_bins=10)

                #print("threshold", threshold)

                left_index = index[feature_vals < threshold] #split left and right data and indices based on the random feature and threshold
                right_index = index[feature_vals >= threshold]
                left = Data[feature_vals < threshold]
                right = Data[feature_vals >= threshold]
                #print(f"Threshold: {threshold}, Left size: {len(left)}, Right size: {len(right)}")
                if len(left) > 0 and len(right) > 0:  # If the length of left and right consider splitting

                    LN = self.random_tree(left, depth + 1, left_index, used_features, random.choice(np.unique(left[:,-1])))
                    RN = self.random_tree(right, depth + 1, right_index, used_features, random.choice(np.unique(right[:,-1])))

                    parent = Node.Node(Data, feature_idx, threshold, data_index = index, left=LN, right=RN, depth=depth)

                    LN.parent = RN.parent = parent

                    return parent

            #split_val = random.randint(1,10)
            #left = Data[:int((split_val/10)*len(Data))]
            #right = Data[int((split_val/10)*len(Data)):]
            #prints for testing, do not remove
            '''
            print(f"Feature {feature_idx} values: min={feature_min}, max={feature_max}")
            print(f"Threshold chosen: {threshold}, Percentage: {percentage}")
            print("Left split size:", len(left), "Right split size:", len(right))
            '''


        #classes = np.unique(y) #Decide the leaf class at random from unique classes in the node
        #random_class = random.choice(classes)
        #print("randomized class: ", random_class)
        correct = 0
        for point in y:
            if point == designated_class:
                correct += 1
        accuracy = correct / len(y) #Calculate accuracy of the node
        #print("Node acc random: ", accuracy)
        #print("Leaf majority class (leaf_value):", leaf_value)
        #print("Class distribution at this node:", np.bincount(y.astype(int)))
        #print(f"Samples: {samples}, Depth: {depth}, Min Samples: {self.min_samples}, Max Depth: {self.Maxdepth}")
        #print(f"Unique labels: {np.unique(y)}")
        return Node.Node(data = Data, value = designated_class, accuracy = accuracy, data_index=index, depth=depth, NoDP=len(Data))


    def random_train(self, Data, indices = None): #takes data and starts the tree creation processes with depth 1
        #print("New tree")
        self.base = self.random_tree(Data, 1, indices)
        self.Node_indexing(self.base, 1)
        #print("Node Base random: ", self.base.right.index, "  ", self.base.right.depth)

    '''def train(self, Data, data_indices = None, worst_index = None, alt_feature = None):
        biased_value = False
        if worst_index is not None:
            biased_value = True
        #print("Worst index contains: ", worst_index)
        self.base = self.yggdrasil(Data, 1, biased_value, worst_index, data_indices, alt_feature)
        self.Node_indexing(self.base, 1)'''

    def train(self, Data, use_subsampling, data_indices=None, worst_index=None, alt_feature=None):
        biased_value = (worst_index is not None)
        self.base = self.yggdrasil(Data, 1, biased_value, use_subsampling, worst_index, data_indices, alt_feature)
        self.Node_indexing(self.base, 1)

    def sum_predictions(self, X, train, worst_indices=None):
        #print("new pred")
        if worst_indices is not None:
            X = X[worst_indices]

        preds = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            preds[i] = self.predict(X[i], self.base, train, i)

        return preds

    def predict(self, X, node, train, index):
        while node.value is None:
            '''print(node.value)
            print(node.threshold)
            print(node.left)
            print(node.right)
            print(node.depth)
            print(node.parent)
            print("End")'''
            '''if train and (node.train_pred_data is None):
                node.train_pred_data = X
                node.train_pred_index = [index]
            elif train and (node.train_pred_data is not None):
                node.train_pred_data = np.vstack((node.train_pred_data, X))
                node.train_pred_index.append(index)'''

            if X[node.feature] > node.threshold:
                node = node.right
            else:
                node = node.left
        return node.value

    def Node_indexing(self, node, index):
        node.index = index
        if node.left is not None and node.right is not None:
            index = self.Node_indexing(node.left, index+1)
            index = self.Node_indexing(node.right, index+1)

        return index


