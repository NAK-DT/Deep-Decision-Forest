from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np


def train_test_split(data, cut): #Splits the dataset into two separate sets based on a procentual cut (0.1-0.9) where the cut represents the percent of data in the training set
    if cut<0.1:
        cut = 0.1
    elif cut > 0.9:
        cut = 0.9
    Train = data[:int(cut*len(data))] #sets the training data from 0 to the procentual cut of the data length converted to integer
    Test = data[int(cut*len(data)):] #Test set becomes the rest of the data
    return Train, Test

def concat(data, true): #adds a column vector onto the end of a matrix
    true = true.reshape(-1,1) #reshape the data to be added to ensure it can be used with hstack
    #print("data shape: ", data.shape, " ground truth shape: ", true.shape)
    ret = data[0] #recreate the data through vstack to make sure everything works correctly (might be redundant)
    for i in range(1,len(data)):
        ret = np.vstack((ret, data[i]))
    return np.hstack((ret, true)) #add the new column at the end before returning

def get_results(Node, array, index_array): #Creates an output for next layer
    if Node.value is None: #if the current node is not a leaf node then iterate through the tree until one is found
        ret_array, upd_index_array = get_results(Node.left, array, index_array) #
        ret_array, upd_index_array = get_results(Node.right, ret_array, upd_index_array)
        return ret_array, upd_index_array

    '''for i in range(len(Node.data)):
        array.append(Node.value)'''
    for i in range(len(Node.data)): #add the datapoint for a leaf node into an array and their indices into another one before returning both arrays
        array.append(Node.value)
        index_array.append(Node.data_index[i])
    return array, index_array


def rearange_data(data, index): #rearanges the data from each tree in a layer to make sure the same predicted datapoints from each tree are in the same row
    correct_order = True
    #print("index shape: ", index[0].shape)
    #print(index[0], "     ", index[1], "     ", index[2], "     ", index[3])
    for i in range(1, len(index)):#Check if the columns are correctly ordered
        if np.any(index[0] != index[i]):
            #print("Index mismatch")
            correct_order = False
            break

    # set the template for correctly ordered data
    template_index = np.arange(len(index[0]))

    if correct_order:
        return data, template_index, index.transpose()



    temp = data[0]
    temp_index = index[0]
    sorting_index = np.argsort(temp_index)
    sorted_temp_index = temp_index[sorting_index]
    correct_position = np.searchsorted(sorted_temp_index, template_index)
    sorted_data = temp[sorting_index][correct_position]
    sorted_index = sorted_temp_index[correct_position]

    #print(" template index: ", template_index)
    rearranged_data = sorted_data #put rearranged data as the data from the first column
    rearranged_index = sorted_index
    #print("rearranged data: ", rearranged_data)
    rearranged_data = rearranged_data.reshape(-1,1) #make sure the rearranged arrays can be modified with hstack
    rearranged_index = rearranged_index.reshape(-1,1)
    for i in range(1, len(index)): #go through every column
        if np.any(template_index != index[i]): #if the current column does not match the template it will be rearranged
            temp_data = data[i] #create temporary variables to hold the data to be rearranged
            temp_index = index[i]
            #print("temp_data shape: ", temp_data.shape)
            #sorted_data = temp_data[template_index].reshape(-1,1)
            #sorted_index = temp_index[template_index].reshape(-1,1)

            sorting_indices = np.argsort(temp_index)  # Get positions of sorted indices
            sorted_temp_index = temp_index[sorting_indices]  # Reorder indices

            # Find the positions of template_index in sorted_temp_index
            correct_positions = np.searchsorted(sorted_temp_index, template_index)

            # Get data in correct order
            sorted_data = temp_data[sorting_indices][correct_positions].reshape(-1, 1)
            sorted_index = sorted_temp_index[correct_positions].reshape(-1, 1)


            #print("Before hstack: ", sorted_data.shape, "    ", rearranged_data.shape)
            rearranged_data = np.hstack((rearranged_data, sorted_data)) #add the rearranged column to the total data
            rearranged_index = np.hstack((rearranged_index, sorted_index))
            #print("After hstack: ", sorted_data.shape, "    ", rearranged_data.shape)
        else:#if the columns is correctly sorted then they will just be added to the data
            rearranged_data = np.hstack((rearranged_data, data[i].reshape(-1, 1)))
            rearranged_index = np.hstack((rearranged_index, index[i].reshape(-1, 1)))

    #print("Shape of rearranged index: ", rearranged_index.shape)
    #print(rearranged_index[:,0], "     ", rearranged_index[:,1], "     ", rearranged_index[:,2], "     ", rearranged_index[:,3])

    return rearranged_data, template_index, rearranged_index

def Majority_voting(Data):
    GT = Data[:, -1]
    MV_data = Data[:, :-1]
    MV_done = []
    for sample in MV_data:
        Major_vote = Counter(sample).most_common(1)[0][0]
        MV_done.append(Major_vote)


    MV_done = np.array(MV_done)
    results = concat(MV_done.transpose(), GT)

    return results

def score(y_pred, y_true, tol=1e-6):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    '''for i in range(len(y_pred)):
         print("sample:", i, "result:", np.isclose(y_pred[i], y_true[i], atol=tol), "y pred:", y_pred[i], "y true:", y_true[i])'''
    return float(np.isclose(y_pred, y_true, atol=tol).mean())


'''def test_predict(Layers, data, weights, use_weights = False):

    prev_data = deepcopy(data)

    for layer in Layers:
        Layer_preds = []
        for tree in layer:
            preds = tree.sum_predictions(prev_data, False)
            Layer_preds.append(preds)

        Layer_preds.append(prev_data[:,-1])
        Layer_preds = np.array(Layer_preds).transpose()
        prev_data = deepcopy(Layer_preds)
    if use_weights:
        MV_test = Majority_voting_weighted(prev_data, weights)
    else:
        MV_test = Majority_voting(prev_data)
    test_score = score(MV_test[:,0], MV_test[:,1])
    print_test_score(MV_test)

    return test_score'''

def test_predict(Layers, data, weights=None, weighted=True):
    y = data[:, -1].copy()
    prev = data                      # L1 expects original features + y

    for layer in Layers:
        # collect this layer's tree outputs (N, T)
        preds = np.column_stack([tree.sum_predictions(prev, False) for tree in layer])
        # feed next layer as [preds, y] ONLY (no extra columns)
        prev = np.column_stack([preds, y])

    Xy = np.column_stack([preds, y])  # preds from last layer + true labels
    if weighted and weights is not None:
        MV = Majority_voting_weighted(Xy, weights)
    else:
        MV = Majority_voting(Xy)
    test_score = score(MV[:, 0], MV[:, 1])
    #print_test_score(MV)
    return test_score

def print_test_score(MV):
    unique_values = np.unique(MV[:,-1])

    for val in unique_values:
        temp_holder = MV[MV[:,-1] == val]
        score = 0
        for i in range(len(temp_holder)):
            if temp_holder[i,0] == temp_holder[i,1]:
                score += 1

        print(f"Correct predictions in class {val} across only class {val} samples:", score/len(temp_holder))
        print(f"Correct predictions for class {val} across entire test dataset:", score/len(MV))

def print_train_score(MV, BP_Type):
    print(f"Running backpropagation {BP_Type}")
    unique_values = np.unique(MV[:,-1])

    for val in unique_values:
        temp_holder = MV[MV[:,-1] == val]
        score = 0
        for i in range(len(temp_holder)):
            if temp_holder[i,0] == temp_holder[i,1]:
                score += 1

        print(f"Correct predictions in class {val} across only class {val} samples:", score/len(temp_holder))
        print(f"Correct predictions for class {val} across entire train dataset:", score/len(MV))

def eval_invid_trees(tree, data):

    preds = tree.sum_predictions(data, False)

    truths = data[:, -1]
    acc_count = 0

    for i in range(len(preds)):
        if preds[i] == truths[i]:
            acc_count += 1

    accuracy = acc_count / len(preds)

    return accuracy


def get_feature_depth_pairs(tree):
    """
    Traverse a decision tree and collect (feature, depth) pairs used at decision nodes.
    """
    pairs = set()

    def traverse(node, depth=0):
        if node is None or node.value is not None:
            return
        pairs.add((node.feature, depth))
        traverse(node.left, depth + 1)
        traverse(node.right, depth + 1)

    traverse(tree.base)
    return pairs


def average_tree_structure_difference(layer):
    """
    Computes the average Jaccard distance of feature-depth usage between all tree pairs in the layer.
    A higher value means trees are more different in structure.
    """
    n = len(layer)
    if n < 2:
        return 0.0

    total_diff = 0
    pair_count = 0

    feature_sets = [get_feature_depth_pairs(tree) for tree in layer]

    for i in range(n):
        for j in range(i + 1, n):
            set1 = feature_sets[i]
            set2 = feature_sets[j]

            union = set1 | set2
            intersection = set1 & set2

            if union:
                jaccard_distance = 1 - len(intersection) / len(union)
                total_diff += jaccard_distance
                pair_count += 1

    return total_diff / pair_count if pair_count > 0 else 0.0


def all_layer_structure_differences(Layers):
    """
    Computes average tree structure differences for all layers in the model.
    Returns a list of differences, one per layer.
    """
    differences = []
    for i, layer in enumerate(Layers):
        diff = average_tree_structure_difference(layer)
        print(f"Layer {i+1} structural difference: {diff:.4f}")
        differences.append(diff)
    return differences

def compute_oob_weights(layer3, layer3_sub_inds, layer2_train_preds, tol=1e-6, alpha = 0.8):
    n = len(layer2_train_preds)
    weights = np.zeros(len(layer3), dtype=np.float64)

    for j, tree in enumerate(layer3):
        in_idx = np.asarray(layer3_sub_inds[j])
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[in_idx] = False
        if not oob_mask.any():
            # fallback if bootstrap sampled everything
            weights[j] = 1e-3
            continue

        oob_X = layer2_train_preds[oob_mask]
        oob_pred = tree.sum_predictions(oob_X, False)
        oob_acc = score(oob_pred, oob_X[:, -1], tol=tol)

        # map accuracy -> weight; smooth to avoid zeros
        # try either of these:
        weights[j] = (oob_acc + 1e-3) ** alpha
        # or a sharper curve:
        # weights[j] = (oob_acc + 1e-3) ** 2
    return weights


def Majority_voting_weighted(Data, weights, tol=1e-6):
    """
    Data: shape (n_samples, n_trees + 1), last col = ground truth
    weights: shape (n_trees,), one weight per tree
    Returns: concat(preds, GT)
    """
    GT = Data[:, -1]
    votes = Data[:, :-1]  # per-tree predicted class (floats)

    # unique classes present in GT (robust to {0,1} or {0,.33,.66,1.})
    classes = np.unique(GT)

    preds = np.empty(len(GT), dtype=np.float64)

    for i in range(len(GT)):
        row = votes[i]
        # sum weights by class
        class_weight = defaultdict(float)
        for j, pj in enumerate(row):
            # find class c that pj matches (within tol)
            # (fast path if exact hit)
            hit = False
            for c in classes:
                if abs(pj - c) <= tol:
                    class_weight[c] += weights[j]
                    hit = True
                    break
            if not hit:
                # if a tree predicts something slightly off-grid, fall back
                # to nearest class by absolute distance
                nearest = classes[np.argmin(np.abs(classes - pj))]
                class_weight[nearest] += weights[j]

        # pick class with max total weight; tie-break to smallest class id for determinism
        preds[i] = max(class_weight.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    return concat(preds, GT)