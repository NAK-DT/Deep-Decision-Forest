import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
from numpy.ma.core import concatenate
from sklearn.model_selection import KFold
from graphviz import Digraph

class DecisionTree:
    def __init__(self, Maxdepth = 4, min_gain = 0.5, min_samples = 2):
        self.Maxdepth = Maxdepth
        self.min_gain = min_gain
        self.min_samples = min_samples
    '''
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.base  # Start from the root node

        if node.value is not None:
            print(" " * depth * 4 + f"Leaf Node: Class={node.value}, Accuracy={node.accuracy:.2f}")
        else:
            print(" " * depth * 4 + f"Feature {node.feature} > {node.threshold:.4f} (Gain={node.information_gain:.4f})")
            if node.left:
                print(" " * depth * 4 + "Left:")
                self.print_tree(node.left, depth + 1)
            if node.right:
                print(" " * depth * 4 + "Right:")
                self.print_tree(node.right, depth + 1)
     '''

    #print decision trees via graphs
    def export_graph(self, node=None, graph=None):
        if node is None:
            node = self.base
            graph = Digraph()

        if node.value is not None:
            graph.node(str(id(node)), f"Leaf: {node.value}\nAcc: {node.accuracy:.2f}")
        else:
            graph.node(str(id(node)), f"Feature {node.feature}\nThreshold: {node.threshold:.2f}\nGain: {node.information_gain:.2f}")
            if node.left:
                graph.edge(str(id(node)), str(id(node.left)), "Left")
                self.export_graph(node.left, graph)
            if node.right:
                graph.edge(str(id(node)), str(id(node.right)), "Right")
                self.export_graph(node.right, graph)

        return graph


    def split(self, data, feature, threshold, data_indices):
        left_data = []
        right_data = []
        left_index = []
        right_index = []
        for i in range(len(data_indices)):
            if data[i][feature] > threshold:
                left_data.append(data[i])
                left_index.append(data_indices[i])
            else:
                right_data.append(data[i])
                right_index.append(data_indices[i])

        left_data = np.array(left_data)
        right_data = np.array(right_data)
        left_index = np.array(left_index)
        right_index = np.array(right_index)
        return left_data, right_data, left_index, right_index

    def entropy(self, y):
        labels = np.unique(y)
        entropy = 0
        for label in labels:
            number = y[y == label]
            amount = len(number)/len(y)
            entropy -= amount * np.log2(amount)

        return entropy

    def information_gain(self, OG, D1, D2):
        OGE = self.entropy(OG)
        D1E = self.entropy(D1)
        D2E = self.entropy(D2)
        D1W = len(D1)/len(OG)
        D2W = len(D2)/len(OG)

        return OGE - D1W*D1E - D2E*D2W

    def find_best_Node_split(self, data, features, data_indices):

        best_split = {"gain": -1, "feature": None, "threshold": None, "LeftData": None, "RightData": None, "LeftIndex": None, "RightIndex": None}
        for feature in range(features):

            feature_vals = data[:, feature]
            treshold = np.unique(feature_vals)
            #print("threshold: ", treshold)
            for tresh in treshold:

                split_left, split_right, left_data_index, right_data_index  = self.split(data, feature, tresh, data_indices)
                #print("Lengths: ", len(split1), "  ", len(split2))
                if len(split_left) and len(split_right):

                    y, y_left, y_right = data[:,-1], split_left[:,-1], split_right[:,-1]
                    gain = self.information_gain(y, y_left, y_right)
                    #print("gain: ", gain)
                    if gain > best_split["gain"]:
                        best_split["gain"] = gain
                        best_split["threshold"] = tresh
                        best_split["feature"] = feature
                        best_split["LeftData"] = split_left
                        best_split["RightData"] = split_right
                        best_split["LeftIndex"] = left_data_index
                        best_split["RightIndex"] = right_data_index

        return best_split

    def leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)

    def yggdrasil(self, Data, depth, data_indices = None):
        X, y = Data[:, :-1], Data[:, -1]
        #print(X.shape, " ", y.shape)
        samples, features = X.shape
        if data_indices is None:
            data_indices = np.arange(samples)

        if samples >= self.min_samples and depth < self.Maxdepth:
            #print("I have reached this")
            best_split = self.find_best_Node_split(Data, features, data_indices)
            #print(best_split["gain"])
            if best_split["gain"] > 0:
                #print("next layer")
                LN = self.yggdrasil(best_split["LeftData"], depth + 1, best_split["LeftIndex"])
                RN = self.yggdrasil(best_split["RightData"], depth + 1, best_split["RightIndex"])

                parent = Node(None, best_split["feature"], best_split['threshold'], best_split['gain'], LN, RN)

                LN.parent = RN.parent = parent

                return parent

        leaf_value = self.leaf_value(y)
        correct = 0
        for point in y:
            if point == leaf_value:
                correct += 1

        accuracy = correct / len(y)
        #print("y: ", len(Data))
        #print("leaf_value: ", leaf_value)
        #print("Node acc trained", accuracy)
        #print("Data: ", Data.shape)
        #print("Number: ", np.sum(y == leaf_value))
        return Node(data=Data, value=leaf_value, accuracy = accuracy, data_index=data_indices)

    def random_tree(self, Data, depth, index = None):
        X, y = Data[:, :-1], Data[:, -1]
        samples, features = X.shape

        if index is None:
            index = np.arange(samples)

        if samples >= self.min_samples and depth < self.Maxdepth:

            feature_idx = random.randint(0, features - 1)
            feature_vals = X[:, feature_idx]

            if feature_vals.dtype == bool:
                    # Handle boolean features
                feature_vals = feature_vals.astype(int)  # Convert to integers

            if len(feature_vals) > 1:
                feature_min, feature_max = feature_vals.min(), feature_vals.max()
                percentage = random.uniform(0.1, 0.9)
                threshold = feature_min + percentage*max((feature_max - feature_min), 1e-6)

                left_index = index[feature_vals <= threshold]
                right_index = index[feature_vals > threshold]
                left = Data[feature_vals <= threshold]
                right = Data[feature_vals > threshold]


            else:
                left, right, left_index, right_index = [], [], [], []

            #split_val = random.randint(1,10)
            #left = Data[:int((split_val/10)*len(Data))]
            #right = Data[int((split_val/10)*len(Data)):]
            #prints for testing, do not remove
            '''
            print(f"Feature {feature_idx} values: min={feature_min}, max={feature_max}")
            print(f"Threshold chosen: {threshold}, Percentage: {percentage}")
            print("Left split size:", len(left), "Right split size:", len(right))
            '''
            if len(left) > 0 and len(right) > 0:


                y, ys1, ys2 = Data[:,-1], left[:,-1], right[:,-1]
                gain = self.information_gain(y, ys1, ys2)

                if gain > 0:
                    LN = self.random_tree(left, depth + 1, left_index)
                    RN = self.random_tree(right, depth + 1, right_index)

                    parent = Node(None, feature_idx, threshold, gain, LN, RN)

                    LN.parent = RN.parent = parent

                    return parent

        classes = np.unique(y)
        random_class = random.choice(classes)
        print("randomized class: ", random_class)
        correct = 0
        for point in y:
            if point == random_class:
                correct += 1
        accuracy = correct / len(y)
        #print("Node acc random: ", accuracy)
        #print("Leaf majority class (leaf_value):", leaf_value)
        #print("Class distribution at this node:", np.bincount(y.astype(int)))
        return Node(data = Data, value = random_class, accuracy = accuracy, data_index=index)

    def biased_tree(self, data, depth, data_indices, index = None):

        X, y = data[:, :-1], data[:, -1]
        samples, features = X.shape
        if index is None:
            index = np.arange(samples)


        biased_mask = np.isin(index, data_indices)
        biased_x, biased_y = X[biased_mask], y[biased_mask]
        biased_data = data[biased_mask]
        biased_index = index[biased_mask]

        if samples <= self.min_samples or depth >= self.Maxdepth or len(np.unique(biased_y)) == 1:
            if len(biased_y)>0:
                value = np.bincount(biased_y.astype(int)).argmax()
                correct = 0
                for point in y:
                    if point == value:
                        correct += 1
                accuracy = correct / len(y)
                return Node(data = data, value = value, accuracy = accuracy, data_index=index )
            else:
                value = np.random.choice(np.unique(y))
                correct = 0
                for point in y:
                    if point == value:
                        correct += 1
                accuracy = correct / len(y)
                return Node(data = data, value = value, accuracy = accuracy, data_index=index)

        best_split = self.find_best_Node_split(biased_data, features, biased_index)

        feature, threshold = best_split["feature"], best_split["threshold"]

        left_mask = X[:,feature] <= threshold
        right_mask = ~left_mask

        left_data, right_data = data[left_mask], data[right_mask]
        left_index, right_index = index[left_mask], index[right_mask]

        left_biased_mask = np.isin(left_index, data_indices)
        right_biased_mask = np.isin(right_index, data_indices)

        left_data_indices, right_data_indices = left_index[left_biased_mask], right_index[right_biased_mask]

        left_child = self.biased_tree(left_data, depth + 1, left_data_indices, left_index)
        right_child = self.biased_tree(right_data, depth + 1, right_data_indices, right_index)

        parent = Node(data = None, feature = feature, threshold = threshold, information_gain = best_split["gain"], left = left_child, right = right_child)

        left_child.parent = right_child.parent = parent

        return parent

    def random_train(self, Data):
        self.base = self.random_tree(Data, 1)

    def train(self, Data, data_indices = None):
        self.base = self.yggdrasil(Data, 1, data_indices)

    def biased_train(self, Data, data_indices):
        self.base = self.biased_tree(Data, 1, data_indices)

    def sum_predictions(self, X):
        preds = []

        for row in X:
            pred = self.predict(row, self.base)
            preds.append(pred)

        preds = np.array(preds)
        #print("Shape of preds: ", np.shape(preds), "\n")
        return preds

    def predict(self, X, node):

        if node.value is not None:
            return node.value

        pred_feature = X[node.feature]
        if pred_feature > node.threshold:
            return self.predict(X, node.left)
        else:
            return self.predict(X, node.right)

    def score(self, y_pred, y_true):
        n_samples = len(y_pred)
        #print(y_pred, " ", y_true)
        correct = np.sum(y_pred == y_true)
        return correct / n_samples

class Node:
    def __init__(self, data = None, feature = None, threshold = None, information_gain = None, left = None, right = None, value = None, parent = None, accuracy = 0, data_index = None):
        self.data = data
        self.feature = feature
        self.threshold = threshold
        self.information_gain = information_gain
        self.left = left
        self.right = right
        self.value = value
        self.parent = parent
        self.accuracy = accuracy
        self.data_index = data_index

def train_test_split(data, cut):
    Train = data[:int(cut*len(data))]
    Test = data[int(cut*len(data)):]
    return Train, Test

def find_worst_feature(node, holder):
    print("Left: ", node.left, " Right: ", node.right)
    if (node.left.value is None) | (node.right.value is None):
        #print("hello")
        if node.left.value is None:
            holder = find_worst_feature(node.left, holder)
        if node.right.value is None:
            holder = find_worst_feature(node.right, holder)
        if node.parent is not None:
            return holder
    else:
        print("Value: ", node.value)
        holder.append(node)
        return holder

    worst_accuracy = 1.1
    worst_node = None
    for Node in holder:
        left_acc = Node.left.accuracy
        print("Left node accuracy: ", left_acc)
        right_acc = Node.right.accuracy
        print("Right node accuracy: ", right_acc)
        average_accuracy = (left_acc + right_acc) / 2
        print("Average accuracy: ", average_accuracy)
        if average_accuracy < worst_accuracy:
            worst_accuracy = average_accuracy
            print("Worst node feature: ", Node.feature)
            worst_node = Node

    Data_from_worst_feature = get_worst_data(worst_node, [])
    Data_array = np.array(Data_from_worst_feature)
    print("Shape of worst data: ", Data_array.shape)
    return worst_node.feature, Data_array

def get_worst_data(Node, list):
    if Node.value is None:
        list = get_worst_data(Node.left, list)
        list = get_worst_data(Node.right, list)
        return list
    for i in range(len(Node.data)):
        list.append(Node.data_index[i])
    #print("Data: ", list)
    return list

def add_layer_data(train, preds):
    initial = train[:,:-1]
    #print(train.shape)
    #print(initial.shape)
    #print(preds.shape)
    preds = preds[:,np.newaxis]
    initial = np.hstack((initial, preds))
    temp = train[:,-1]
    temp = temp[:,np.newaxis]
    initial = np.hstack((initial, temp))
    #print(initial.shape)
    return initial

def add_to_test(tree, train, test):
    pred_data = test[:,:-1]
    tree.train(train)
    predictions = tree.sum_predictions(pred_data)
    predictions = predictions[:,np.newaxis]
    temp = test[:,-1]
    temp = temp[:,np.newaxis]
    new_test = np.hstack((pred_data, predictions))
    new_test = np.hstack((new_test, temp))
    return new_test

def get_results(Node, array, index_array): #Creates an output for next layer
    if Node.value is None:
        ret_array, upd_index_array = get_results(Node.left, array, index_array)
        ret_array, upd_index_array = get_results(Node.right, ret_array, upd_index_array)
        return ret_array, upd_index_array

    '''for i in range(len(Node.data)):
        array.append(Node.value)'''
    for i in range(len(Node.data)):
        array.append(Node.value)
        index_array.append(Node.data_index[i])
    return array, index_array

def concat(data, true):
    true = true.reshape(-1,1)
    print("data shape: ", data.shape, " ground truth shape: ", true.shape)
    ret = data[0]
    for i in range(1,len(data)):
        ret = np.vstack((ret, data[i]))
    return np.hstack((ret, true))

def find_multi_features(layers):
    ret_features = []
    for tree in layers:
        ret_feat = find_worst_feature(tree.base, [])
        if ret_feat not in ret_features:
            ret_features.append(ret_feat)

    return ret_features

def rearange_data(data, index):
    correct_order = True
    print("index shape: ", index[0].shape)
    print(index[0], "     ", index[1], "     ", index[2], "     ", index[3])
    for i in range(1, len(index)):
        if np.any(index[0] != index[i]):
            print("Index mismatch")
            correct_order = False

    if correct_order:
        return data, index[0]

    template_index = index[0]
    template_data = data[0]
    rearranged_data = template_data
    rearranged_data = rearranged_data.reshape(-1,1)
    rearranged_index = template_index.reshape(-1,1)
    for i in range(1, len(index)):
        if np.any(template_index != index[i]):
            temp_data = data[i]
            temp_index = index[i]
            print("temp_data shape: ", temp_data.shape)
            #sorted_data = temp_data[template_index].reshape(-1,1)
            #sorted_index = temp_index[template_index].reshape(-1,1)

            sorting_indices = np.argsort(temp_index)  # Get positions of sorted indices
            sorted_temp_index = temp_index[sorting_indices]  # Reorder indices

            # Find the positions of template_index in sorted_temp_index
            correct_positions = np.searchsorted(sorted_temp_index, template_index)

            # Get data in correct order
            sorted_data = temp_data[sorting_indices][correct_positions].reshape(-1, 1)
            sorted_index = sorted_temp_index[correct_positions].reshape(-1, 1)


            print("Before hstack: ", sorted_data.shape, "    ", rearranged_data.shape)
            rearranged_data = np.hstack((rearranged_data, sorted_data))
            rearranged_index = np.hstack((rearranged_index, sorted_index))
            print("After hstack: ", sorted_data.shape, "    ", rearranged_data.shape)
        else:
            rearranged_data = np.hstack((rearranged_data, data[i].reshape(-1, 1)))
            rearranged_index = np.hstack((rearranged_index, index[i].reshape(-1, 1)))

    print("Shape of rearranged index: ", rearranged_index.shape)
    print(rearranged_index[:,0], "     ", rearranged_index[:,1], "     ", rearranged_index[:,2], "     ", rearranged_index[:,3])

    return rearranged_data, template_index, rearranged_index




data = pd.read_csv('mushroomfinal (1).csv')
print(data.info())
data = np.array(data)
np.random.shuffle(data)
#print("Data: ", data)
train, test = train_test_split(data, 0.7)

'''
#Code for testing purpose do not yet remove
for min_samples in [2, 10, 20, 50]:
    print(f"Testing with min_samples={min_samples}")
    rand = DecisionTree(Maxdepth=4, min_gain=0.5, min_samples=min_samples)
    rand.random_train(train)  # Train with your random tree logic

    # Evaluate the tree on the test set
    predictions = rand.sum_predictions(test[:, :-1])
    accuracy = rand.score(predictions, test[:, -1])
    print(f"Accuracy with min_samples={min_samples}: {accuracy}\n")

'''
#Code for mutliple trees
unique, counts = np.unique(data[:, -1], return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))
layer1 = []
temp = []
temp_preds = []
data_indeces = []
layer1_acc = []
all_layers = []
'''
for i in range(4):
    rand = DecisionTree()
    rand.random_train(train)
    layer1.append(rand)
    temp.append(get_results(rand.base, []))
    preds = rand.sum_predictions(test[:, :-1])
    temp_preds.append(preds)
    accuracy = rand.score(preds, test[:, -1])
    layer1_acc.append(accuracy)

print("mean accuracy Layer 1: ", np.mean(layer1_acc))
temp_preds = np.array(temp_preds)
temp = np.array(temp)
new_data = concat(temp, train[:,-1])
new_preds = concat(temp_preds, test[:,-1])

layer2 = []
layer2_acc = []

for i in range(4):
    trained = DecisionTree()
    trained.train(new_data)
    layer2.append(trained)
    layer2_preds = trained.sum_predictions(new_preds[:, :-1])
    layer2_accuracy = trained.score(layer2_preds, new_preds[:, -1])
    layer2_acc.append(layer2_accuracy)
    print(f"layer 2 tree {i+1} accuracy: {layer2_accuracy}")

print("mean accuracy Layer 2: ", np.mean(layer2_acc))

retrain_list = find_multi_features(layer2)
print(f"Tree(s) to retrain: {retrain_list}")

'''

#Code for testing re training function for a single tree in layer 2
for i in range(4):
    rand = DecisionTree()
    rand.random_train(train)
    layer1.append(rand)
    preds = rand.sum_predictions(test[:, :-1])
    temp_preds.append(preds)
    accuracy = rand.score(preds, test[:, -1])
    layer1_acc.append(accuracy)
    temp_data, data_index = get_results(rand.base, [], [])
    temp.append(temp_data)
    data_indeces.append(data_index)
data_indeces = np.array(data_indeces)
temp = np.array(temp)
print(temp.shape, "   ", data_indeces.shape)
temp, template_indeces, data_indeces = rearange_data(temp, data_indeces)
print(temp.shape,"   ", data_indeces.shape)

layer1_acc = np.array(layer1_acc)
#print("prediction shapes: ", np.shape(temp_preds))
#print("training shapes: ", np.shape(temp))
print("Mean accuracy layer 1 (pre retrain): ", layer1_acc.mean())
#print(layer1[0].base.left.left.parent.parent.information_gain == layer1[0].base.information_gain == layer1[0].base.right.parent.information_gain)
#print(layer1[0].base.parent)
temp_preds = np.array(temp_preds)
temp = np.array(temp)
new_data = concat(temp, train[:,-1])
print("Data after changes: ", new_data)
new_preds = concat(temp_preds.transpose(), test[:,-1])

#print("shape of new data: ", new_data.shape, "\n")
#print("shape of new preds: ", new_preds.shape, "\n")

print("total Data: ", np.sum(new_data[:,-1]==True))
layer2 = DecisionTree()
layer2.train(new_data, template_indeces)
final_preds = layer2.sum_predictions(new_preds[:, :-1])
accuracyL2 = layer2.score(final_preds, new_preds[:, -1])

print("Layer 2 Accuracy pre retraining: ", accuracyL2)
ret_feat, worst_data_indexes= find_worst_feature(layer2.base, [])
print(ret_feat)
layer1[ret_feat].biased_train(train, worst_data_indexes)
retrain_preds = layer1[ret_feat].sum_predictions(test[:, :-1])
temp_preds[ret_feat] = retrain_preds
new_score = layer1[ret_feat].score(retrain_preds, test[:, -1])
layer1_acc[ret_feat] = new_score
print("Layer 1 accuracy post retraining: ", np.mean(layer1_acc))
results, indices = get_results(layer1[ret_feat].base, [], [])
#temp2 = np.array(temp2)
#print("Shape of data: ", temp2, " To assign to: ", temp[:,0])
#temp[:,ret_feat] = temp2, data_indeces[ret_feat] = temp3

results = np.array(results).reshape(-1,1)
indices = np.array(indices)
temp[:,ret_feat] = results.squeeze()
data_indeces[:,ret_feat] = indices
print("shape of temp: ", temp.shape)

print("template_indeces before update: ", template_indeces)
temp, template_indeces, data_indeces = rearange_data(temp.transpose(), data_indeces.transpose())
print("template_indeces after update: ", template_indeces)
print("temp shape: ", temp.shape)

back_data = concat(temp, train[:,-1])
print("Back data shape: ", back_data.shape)
print("temp preds shape: ", temp_preds.shape, " test ground truth shape: ", test[:,-1].shape)
new_preds = concat(temp_preds.transpose(), test[:,-1])
layer2.train(back_data, template_indeces)
final_preds = layer2.sum_predictions(new_preds[:, :-1])
accuracyL2 = layer2.score(final_preds, new_preds[:, -1])

print("Layer 2 Accuracy post retraining: ", accuracyL2)

#Code ends here

#print(new_data)
'''
tree = DecisionTree()
pre = cross_val(tree, train)
new_data = add_layer_data(train, pre)
new_test = add_to_test(tree, train, test)
tree2 = DecisionTree()
tree2.train(new_data)
pred2 = tree2.sum_predictions(new_test[:,:-1])
print(tree.score(pred2, new_test[:,-1]))'''
print('Done')

