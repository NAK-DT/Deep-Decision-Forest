import heapq
import random
from copy import deepcopy
import numpy as np
import Trees
from collections import Counter, defaultdict
import modify_data
import strategy
from modify_data import print_train_score


def get_Biased_results(worst_data, compare_list, mid_layer_check = None):
    Biased_data = []
    Biased_classes = []
    i = 0
    for tup in worst_data:
        for item in compare_list:
            #print("tup: ", tup[0], "item: ", item)
            if tup[0] == item[0]:
                if tup[1] != item[1]:
                    if mid_layer_check:
                        if item[1] == mid_layer_check[i]:
                            Biased_data.append(tup[0])
                            Biased_classes.append([item[2], item[3]])
                            break

                        else:
                            Biased_data.append(tup[0])
                            Biased_classes.append([tup[2], tup[3]]) #<----- ground truth label isntead of binary
                            break
                    else:
                        if item[1] == tup[-1][-1]:
                            Biased_data.append(tup[0])
                            Biased_classes.append([item[2], item[3]])
                            #print("Found biased point")
                            break

                        elif tup[1] == tup[-1][-1]:
                            Biased_data.append(tup[0])
                            Biased_classes.append([tup[2], tup[3]]) #<---- ground truth label
                            break

                        else:
                            break
        i = i + 1

    return Biased_data, Biased_classes

'''def get_worst_data(Node, temp_list):
    if Node.value is None:
        temp_list = get_worst_data(Node.left, temp_list)
        temp_list = get_worst_data(Node.right, temp_list)
        return temp_list
    for i in range(len(Node.data)):
        temp_list.append([Node.data_index[i], Node.data[i], Node.value])
    #print("Data: ", list)
    return temp_list'''

def get_worst_data(Node, data, index, temp_list, side = None, Wanted_threshold = None, Threshold = None):

    if index is None:
        print(f"Node value: {Node.value}, Node index: {index}, Node data: {data}, Node feature: {Node.feature}")

    if Node.value is not None:
        for i in range(len(index)):
            temp_list.append([index[i], Node.value, side, Wanted_threshold, data[i]])
        return temp_list
    else:

        left_data = []
        right_data = []
        left_index = []
        right_index = []
        if Threshold is not None:
            thresh = Threshold
        else:
            thresh = Node.threshold

        for i in range(len(index)):
            if data[i][Node.feature] < thresh:
                left_data.append(data[i])
                left_index.append(index[i])
            else:
                right_data.append(data[i])
                right_index.append(index[i])
        if len(left_index) > 0:
            if Threshold is not None:
                Wanted_threshold = Threshold
                side = 'L'
            temp_list = get_worst_data(Node.left, left_data, left_index, temp_list, side = side, Wanted_threshold = Wanted_threshold)
        if len(right_index) > 0:
            if Threshold is not None:
                Wanted_threshold = Threshold
                side = 'R'
            temp_list = get_worst_data(Node.right, right_data, right_index, temp_list, side = side, Wanted_threshold = Wanted_threshold)


        return temp_list

def alternate_split(Node, data, index, temp_list, alt, side = None, Wanted_threshold = None, Threshold = None):
    if alt:
        Wanted_threshold = Threshold
        right_side = 'R'
        left_side = 'L'
        left_data = []
        right_data = []
        right_index = []
        left_index = []

        if Node.value is not None:
            for i in range(len(index)):
                temp_list.append([index[i], Node.value, side, Wanted_threshold])
            return temp_list
        else:

            for datapoint in data:
                if datapoint[-1][Node.feature] < Threshold:
                    right_data.append(datapoint[-1])
                    right_index.append(datapoint[0])
                else:
                    left_data.append(datapoint[-1])
                    left_index.append(datapoint[0])
            if len(right_index) > 0:
                temp_list = alternate_split(Node.right, right_data, right_index, temp_list, False, side = right_side, Wanted_threshold = Wanted_threshold)
            if len(left_index) > 0:
                temp_list = alternate_split(Node.left, left_data, left_index, temp_list, False, side = left_side, Wanted_threshold = Wanted_threshold)

            return temp_list


    else:
        if Node.value is not None:
            for i in range(len(index)):
                temp_list.append([index[i], Node.value, side, Wanted_threshold])
            return temp_list
        else:
            left_data = []
            right_data = []
            left_index = []
            right_index = []
            for i in range(len(index)):
                if data[i][Node.feature] <= Node.threshold:
                    left_data.append(data[i])
                    left_index.append(index[i])
                else:
                    right_data.append(data[i])
                    right_index.append(index[i])
            if len(left_index) > 0:
                temp_list = alternate_split(Node.left, left_data, left_index, temp_list, False, side = side, Wanted_threshold = Wanted_threshold)
            if len(right_index) > 0:
                temp_list = alternate_split(Node.right, right_data, right_index, temp_list, False, side = side, Wanted_threshold = Wanted_threshold)

            return temp_list


def acquire_split_nodes(node, Node_collection):
    if node.feature is not None:
        Node_collection.append(node)
    #print(node.value)

    if node.left is not None:
        if node.left.value is None:
            Node_collection = acquire_split_nodes(node.left, Node_collection)


    if node.right is not None:
        if node.right.value is None:
            Node_collection = acquire_split_nodes(node.right, Node_collection)

    return Node_collection

def sum_classes(data, classes):
    ret_data = []
    ret_classes = []
    for i, sett in enumerate(classes):
        cnt = Counter(sett)
        most_common = cnt.most_common()
        if len(most_common) == 1:
            ret_data.append(data[i])
            ret_classes.append(most_common[0][0])

    return ret_data, ret_classes


def sum_data(votes, thresholds, cap=None):

    thresholds = sorted(thresholds)
    prev_below = {t: (None if i == 0 else thresholds[i-1]) for i, t in enumerate(thresholds)}

    out_idx, out_targ = [], []

    for idx, pairs in votes.items():
        if not pairs:
            continue

        r_vals_all = [t for (s, t) in pairs if s == 'R']
        l_vals_all = [t for (s, t) in pairs if s == 'L']

        # Both-sides bounds
        min_R = min(r_vals_all) if r_vals_all else None   # tightest lower bound
        max_L = max(l_vals_all) if l_vals_all else None   # tightest upper bound

        # No signal
        if min_R is None and max_L is None:
            continue

        # Both sides present
        if min_R is not None and max_L is not None:
            # conflict if bounds touch or cross
            if min_R >= max_L:
                continue

            target = min_R
            # Count only votes consistent with the valid zone [min_R, max_L)
            weight = sum(
                1 for (s, t) in pairs
                if (s == 'R' and t <= max_L) or (s == 'L' and t >= min_R)
            )

        # R-only: use the largest R
        elif min_R is not None:
            largest_R = max(r_vals_all)
            target = largest_R
            weight = len(r_vals_all)

        # L-only: use the smallest L, then map to previous threshold (or 0.0)
        else:
            smallest_L = min(l_vals_all)
            below = prev_below.get(smallest_L)
            target = 0.0 if below is None else below
            weight = len(l_vals_all)

        if cap is not None:
            weight = min(weight, cap)
        if weight <= 0:
            continue

        out_idx.extend([idx] * int((weight/len(thresholds))))
        out_targ.extend([target] * int((weight/len(thresholds))))

    return out_idx, out_targ




def find_best_improvement_single(
    Layers, Data, Prev_Layer_data, current_acc,
    include_random_trees, distribution, number_of_features,
    maximum_tries, worst_tree, number_threshs, possible_classes, L3_sub_inds
):
    from collections import defaultdict
    import heapq
    from copy import deepcopy

    bad_features = []

    # always-defined return fields
    accepted_count = 0
    accepted_slots = []

    best_improvement  = 0.0
    best_accuracy     = 0.0
    best_layers       = None
    best_layers_data  = None
    best_feature      = None
    single_L3_inds = None
    single_weights = None

    retrain_indices = []
    retrain_classes = []
    tries = 0

    # build feature/node map
    feature_split_Nodes, Existing_features = sortSplitNode(Layers, depth=0)

    tries_to_find_data = 0

    while tries < maximum_tries:
        # choose a scoring strategy
        indexed_arr, _, __, Shap = strategy.average_node_depth(
            Existing_features, feature_split_Nodes
        )

        #indexed_arr, _, __, Shap = strategy.su_computation(feature_split_Nodes, len(Layers[-1]))

        # if we have nothing to rank on, abort this try early
        if not indexed_arr:
            tries += 1
            continue

        # gather retrain candidates (indices/classes) until we have some
        while not retrain_indices or not retrain_classes:

            # pick worst (or best) features
            if Shap:
                worst_feats = heapq.nsmallest(number_of_features, indexed_arr, key=lambda x: x[1])
            else:
                worst_feats = heapq.nlargest(number_of_features, indexed_arr, key=lambda x: x[1])

            if not worst_feats:
                tries += 1
                break

            worst_features = sorted([idx for idx, _ in worst_feats])

            Layers_Nodes = featCount(worst_features, feature_split_Nodes, depth=0)

            thresholds = [(i + 1) / (number_threshs + 1) for i in range(number_threshs+1)]
            for Feat_index, Layer_Nodes in enumerate(Layers_Nodes):
                data_indices = []
                data_classes = []
                votes = defaultdict(list)

                for node_tuple in Layer_Nodes:
                    for t in thresholds:
                        # use prediction-path caches
                        worst_data = get_worst_data(
                            node_tuple[1],
                            node_tuple[1].data,
                            node_tuple[1].data_index,
                            [],
                            Threshold=t
                        )

                        alt_data = alternate_split(node_tuple[1], worst_data, None, [], True, Threshold=t)
                        Bia_data, Bia_classes = get_Biased_results(worst_data, alt_data)

                        for idx, (side, thr_val) in zip(Bia_data, Bia_classes):
                            votes[idx].append((side, thr_val))

                # aggregate votes across thresholds, cap per-index contributions
                data_indices, data_classes = sum_data(votes, thresholds, cap=6)

                if not data_indices:
                    # no usable data for this feature → drop it and retry
                    retrain_indices = []
                    retrain_classes = []
                    indexed_arr = [item for item in indexed_arr if item[0] != worst_features[Feat_index]]
                    tries_to_find_data += 1
                    break  # break Feat loop → recompute worst_features

                retrain_indices.append(data_indices.copy())
                retrain_classes.append(data_classes.copy())

            if tries_to_find_data >= 5:
                tries += 1
                tries_to_find_data = 0
                # clear and retry outer loop
                retrain_indices = []
                retrain_classes = []
                break

        # if we didn’t assemble *both* indices and classes, retry
        if not (retrain_indices and retrain_classes):
            retrain_indices = []
            retrain_classes = []
            tries += 1
            continue

        # train and evaluate this single-layer modification
        temp_layers, temp_data, temp_acc, temp_feature, accepted_count, accepted_slots, single_L3_inds, single_weights = train_single(
            Layers, worst_features, Data, Prev_Layer_data,
            include_random_trees, retrain_indices, retrain_classes,
            distribution, worst_tree, L3_sub_inds
        )

        if temp_acc - current_acc >= best_improvement:
            # refresh structures for next rounds and store best
            feature_split_Nodes, Existing_features = sortSplitNode(temp_layers, depth=0)
            _ = featCount(worst_features, feature_split_Nodes, depth=0)  # keeps behavior, even if unused

            best_layers       = temp_layers
            best_layers_data  = temp_data
            best_accuracy     = temp_acc
            best_improvement  = temp_acc - current_acc
            best_feature      = temp_feature
            print("found improvement shap single: ", best_feature)
            break
        else:
            # retry with a fresh attempt
            retrain_indices = []
            retrain_classes = []
            tries += 1

    print("best returned accuracy shap single: ", best_accuracy)
    return best_layers, best_layers_data, best_accuracy, best_feature, accepted_count, accepted_slots, single_L3_inds, single_weights

def find_best_improvement_propagate(
    Layers, Data, current_acc, depth,
    feat_holder, indices_holder, class_holder,
    include_random_trees, distribution, number_of_features, featu,
    maximum_tries, worst_tree, number_threshs, possible_classes, L3_sub_inds,
    Extra_indices=None, Retrain_datasets=None, old_indices=None, old_classes=None
):
    from collections import defaultdict
    import heapq
    from copy import deepcopy

    # ---- safe defaults
    if Extra_indices is None:
        Extra_indices = []
    if Retrain_datasets is None:
        Retrain_datasets = []

    best_improvement = 0.0
    best_accuracy   = 0.0
    best_layers     = None
    best_layers_data= None
    best_feats      = None
    propagate_L3_inds = None
    propagate_weights = None

    retrain_indices = []
    retrain_classes = []
    find_data_tries = 0
    tries = 0

    # always-defined return values
    accepted_count = 0
    accepted_slots = []

    feature_split_Nodes, Existing_features = sortSplitNode(Layers, depth, featu)

    while tries < maximum_tries:
        indexed_arr, biased_shapley_indices, biased_shapley_classes, Shap = strategy.average_node_depth(
            Existing_features, feature_split_Nodes, old_indices, old_classes
        )

        #indexed_arr, biased_shapley_indices, biased_shapley_classes, Shap = strategy.su_computation(feature_split_Nodes, len(Layers[-1]), old_indices, old_classes)

        while retrain_indices == [] or retrain_classes == []:
            if not indexed_arr:   # nothing to pick
                break

            if Shap:
                worst_feats = heapq.nsmallest(number_of_features, indexed_arr, key=lambda x: x[1])
            else:
                worst_feats = heapq.nlargest(number_of_features, indexed_arr, key=lambda x: x[1])

            worst_features = sorted([idx for idx, _ in worst_feats])
            Layers_Nodes = featCount(worst_features, feature_split_Nodes, depth)

            thresholds = [(i + 1) / (number_threshs + 1) for i in range(number_threshs+1)]

            for Feat_index, Layer_Nodes in enumerate(Layers_Nodes):
                data_indices = []
                data_classes = []
                votes = defaultdict(list)

                for node_tuple in Layer_Nodes:
                    for t in thresholds:
                        worst_data = get_worst_data(
                            node_tuple[1],
                            node_tuple[1].data,
                            node_tuple[1].data_index,
                            [],
                            Threshold=t
                        )

                        if biased_shapley_indices is not None:
                            temp_holder = []
                            new_classes = []
                            for tup in worst_data:
                                if np.isin(tup[0], biased_shapley_indices):
                                    temp_index = biased_shapley_indices.index(tup[0])
                                    temp_test = biased_shapley_classes[temp_index]
                                    new_classes.append(temp_test)
                                    temp_holder.append(tup)
                            worst_data = temp_holder
                        else:
                            new_classes = None

                        alt_data = alternate_split(node_tuple[1], worst_data, None, [], True, Threshold=t)
                        Bia_data, Bia_classes = get_Biased_results(worst_data, alt_data, new_classes)

                        for idx, (side, ThresholdVal) in zip(Bia_data, Bia_classes):
                            votes[idx].append((side, ThresholdVal))

                data_indices, data_classes = sum_data(votes, thresholds, cap=6)

                if not data_indices:
                    # no data for this feature → drop it and try again
                    retrain_indices = []
                    retrain_classes = []
                    indexed_arr = [item for item in indexed_arr if item[0] != worst_features[Feat_index]]
                    find_data_tries += 1
                    break  # try a new worst_features set

                retrain_indices.append(data_indices.copy())
                retrain_classes.append(data_classes.copy())

            if find_data_tries == 5:
                break  # give up trying to assemble data this round

        # reset for the next outer try
        find_data_tries = 0

        if retrain_indices and retrain_classes:
            feat_holder.insert(0, worst_features)
            indices_holder.insert(0, retrain_indices)
            class_holder.insert(0, retrain_classes)

            # ----- prepare lower-layer evaluation ONLY if we actually got a mask
            if depth != len(Layers) and (biased_shapley_indices is not None) and (biased_shapley_classes is not None):
                layer_data = deepcopy(Data[depth - 1])
                # guard shape/type
                try:
                    eval_data = layer_data[biased_shapley_indices]
                    eval_data[:, -1] = biased_shapley_classes
                    eval_Layer = np.copy(Layers[depth - 1])

                    worst_accuracy = 1.0
                    worst_layer_tree = None

                    for idx, tree in enumerate(eval_Layer):
                        tree_accuracy = modify_data.eval_invid_trees(tree, eval_data)
                        if (tree_accuracy < worst_accuracy) and (not np.isin(idx, feat_holder[1])):
                            worst_accuracy = tree_accuracy
                            worst_layer_tree = idx

                    # if nothing identified, keep a placeholder to avoid None surprises
                    if worst_layer_tree is None:
                        worst_layer_tree = 0

                    worst_tree.insert(0, worst_layer_tree)
                    Retrain_datasets.append(eval_data)
                    Extra_indices.append(biased_shapley_indices)
                except Exception:
                    # if indexing fails, skip creating eval dataset for this depth
                    pass

            # ----- descend or train
            if depth == 2:
                layers, layer_data, pot_acc, feats, accepted_count, accepted_slots, propagate_L3_inds, propagate_weights = train_propagate(
                    Layers, feat_holder, Data[0], Data[1:],
                    indices_holder, class_holder,
                    include_random_trees, distribution,
                    worst_tree, Retrain_datasets, Extra_indices, L3_sub_inds
                )
                new_layers = deepcopy(layers)
            else:
                layers, layer_data, pot_acc, feats, accepted_count, accepted_slots, propagate_L3_inds, propagate_weights = find_best_improvement_propagate(
                    Layers, Data, current_acc, depth - 1,
                    feat_holder, indices_holder, class_holder,
                    include_random_trees, distribution, number_of_features,
                    worst_features, maximum_tries, worst_tree, number_threshs, possible_classes, L3_sub_inds,
                    Retrain_datasets, Extra_indices, retrain_indices, retrain_classes
                )
                new_layers = deepcopy(layers)

            # ----- keep best
            if pot_acc - current_acc >= best_improvement:
                feature_split_Nodes, Existing_features = sortSplitNode(new_layers, depth, featu)
                Layers_Nodes = featCount(worst_features, feature_split_Nodes, depth)

                best_layers      = layers
                best_layers_data = layer_data
                best_improvement = pot_acc - current_acc
                best_accuracy    = pot_acc
                best_feats       = deepcopy(feats)
                print("found improvement shap prop: ", best_feats)
                break
            else:
                retrain_indices = []
                retrain_classes = []
                tries += 1

            # pop the state we pushed
            feat_holder.pop(0)
            indices_holder.pop(0)
            class_holder.pop(0)
        else:
            retrain_indices = []
            retrain_classes = []
            tries += 1

    print("Best accuracy after testing shap propagate: ", best_accuracy)
    return best_layers, best_layers_data, best_accuracy, best_feats, accepted_count, accepted_slots, propagate_L3_inds, propagate_weights


def sortSplitNode(Layers, depth, featu=None):
    if featu is None:
        featu = []

    feature_split_Nodes = []
    Existing_features = []

    feature_set = set(Existing_features)  # Use a set for faster lookups

    for treedex, tree in enumerate(Layers[depth - 1]):
        temp_node_collection = acquire_split_nodes(tree.base, [])

        for node in temp_node_collection:
            if node.feature not in feature_set:
                if featu == [] or np.isin(treedex, featu):
                    Existing_features.append(node.feature)
                    feature_set.add(node.feature)  # Keep set updated

            feature_split_Nodes.append([treedex, node, node.feature])  # Store the node info
    return feature_split_Nodes, Existing_features

def featCount(worst_features, feature_split_Nodes, depth):
    Layers_Nodes = []
    feature_count = []

    for feat in worst_features:
        temp_nodes = [node_info.copy() for node_info in feature_split_Nodes if node_info[2] == feat]
        present_trees = []
        pres_counter = 0
        for collection in temp_nodes:
            if not np.isin(collection[0], present_trees):
                present_trees.append(collection[0])
                pres_counter += 1
        feature_count.append([feat, pres_counter])

        Layers_Nodes.append(temp_nodes)
    '''for item in feature_count:
        print(f"Feature {item[0]} was present in {item[1]} trees at depth {depth}.")'''
    return Layers_Nodes




def train_propagate(Old_Layers, feature, Data, Change_data, biased_data, biased_classes,
                    include_random_trees, distribution, worst_tree, Retraining_datasets, retraining_indices, L3_sub_inds, use_weights = True):

    accepted_count = 0
    accepted_slots = []

    # ----- L1 update (propagated) -----
    OG_features = feature[0]
    OG_classes  = biased_classes[0]
    OG_indices  = biased_data[0]

    pot_layer1 = Old_Layers[0].copy()
    Layer1_data_copy = deepcopy(Change_data[0])

    for i in range(len(OG_features)):
        pot_layer1_tree = Trees.DecisionTree()
        pot_layer1_tree.train(Data, False, None, OG_indices[i], OG_classes[i])
        pot_layer1_trainpreds = pot_layer1_tree.sum_predictions(Data, True)

        pot_layer1[OG_features[i]] = pot_layer1_tree
        Layer1_data_copy[:, OG_features[i]] = pot_layer1_trainpreds.copy()

    prev_data = np.array(Layer1_data_copy)
    layers = [pot_layer1]
    ret_data = [Layer1_data_copy]

    # ----- L2 update (propagated) -----
    for i in range(1, len(Old_Layers) - 1):
        layer_holder = Old_Layers[i].copy()
        layer_data   = Change_data[i].copy()

        retrain_features = feature[i]
        retrain_indices  = biased_data[i]
        retrain_classes  = biased_classes[i]

        idx_ptr = 0
        for j in range(len(retrain_features)):
            indices = retrain_indices[idx_ptr].copy()
            data_to_train = np.array(prev_data[indices])

            new_tree = Trees.DecisionTree()
            new_tree.train(data_to_train, False, indices, None, retrain_classes[idx_ptr])
            idx_ptr += 1

            tree_preds = new_tree.sum_predictions(prev_data.copy(), True)
            layer_holder[retrain_features[j]]   = new_tree
            layer_data[:, retrain_features[j]]  = tree_preds

        # replace the one worst L2 tree for this layer
        pot_new_layer_tree = Trees.DecisionTree()
        pot_new_layer_tree.train(Retraining_datasets[i-1], True, retraining_indices[i-1])
        layer_holder[worst_tree[i-1]] = pot_new_layer_tree
        pot_new_tree_preds = pot_new_layer_tree.sum_predictions(prev_data.copy(), True)
        layer_data[:, worst_tree[i-1]] = np.copy(pot_new_tree_preds)

        prev_data = layer_data
        layers.append(layer_holder)
        ret_data.append(layer_data)

    # ================= L3 RETRAIN WITH MULTI-SPLIT VALIDATION =================
    pot_final_layer      = deepcopy(Old_Layers[-1])       # list of L3 trees
    pot_final_layer_data = deepcopy(Change_data[-1])      # [n_rows, n_trees] L3 outputs for ALL rows

    prev_data_arr = np.asarray(prev_data)                 # L2 outputs (features for L3)
    y_true_all    = Data[:, -1]

    K = 5
    rng = np.random.RandomState(42)

    # stratified K splits
    val_splits = []
    for _ in range(K):
        val_idx_list = []
        trn_idx_list = []
        for cls in np.unique(y_true_all):
            cls_idx = np.where(y_true_all == cls)[0]
            rng.shuffle(cls_idx)
            n_val = max(1, int(0.20 * len(cls_idx)))
            val_idx_list.append(cls_idx[:n_val])
            trn_idx_list.append(cls_idx[n_val:])
        val_idx = np.concatenate(val_idx_list)
        trn_idx = np.concatenate(trn_idx_list)
        # L3 view on this split
        L3_val = pot_final_layer_data[val_idx, :].copy()
        y_val = y_true_all[val_idx]
        val_splits.append({
            "val_idx": val_idx,
            "trn_idx": trn_idx,
            "L3_val": L3_val,
            "y_val": y_val
        })
    # --- BASELINES ---
    if use_weights:
        base_weights = modify_data.compute_oob_weights(pot_final_layer, L3_sub_inds, prev_data_arr)  # len=T
        base_weights_ext = np.append(base_weights, 0.0)  # pad for label col
        base_scores = []
        for sp in val_splits:
            base_full = np.column_stack([sp["L3_val"], sp["y_val"]])  # (N_val, T+1)
            MV = modify_data.Majority_voting_weighted(base_full, base_weights_ext)
            base_scores.append(modify_data.score(MV[:, 0], MV[:, 1]))
    else:
        base_scores = [modify_data.score(
            modify_data.Majority_voting(np.column_stack([sp["L3_val"], sp["y_val"]]))[:, 0],
            modify_data.Majority_voting(np.column_stack([sp["L3_val"], sp["y_val"]]))[:, 1]
        ) for sp in val_splits]



        #---- iterate only the requested L3 slots
    for slot in worst_tree[-1] if isinstance(worst_tree, list) and isinstance(worst_tree[-1],
                                                                              (list, np.ndarray)) else worst_tree:
        best_mean_delta = float('-inf')
        best_tree = None
        best_preds_all = None
        best_sub_inds = None

        # knobs
        M_CANDIDATES = 50
        BAG_FRAC = 0.63
        EPS = 0  # allow tiny no-worse changes

        for _ in range(M_CANDIDATES):
            # (A) rotate which split we source training indices from to increase diversity
            if _ >= M_CANDIDATES:
                BAG_FRAC = 0.63
            split_idx = np.random.randint(len(val_splits))
            trn_pool = val_splits[split_idx]["trn_idx"]
            bag_sz = max(1, int(len(trn_pool) * BAG_FRAC))
            if _ >= M_CANDIDATES:
                bag_idx = np.random.choice(trn_pool, bag_sz, replace=True)
            else:
                bag_idx = np.random.choice(trn_pool, size=bag_sz, replace=True)

            cand = Trees.DecisionTree()
            cand.train(prev_data_arr[bag_idx], True, bag_idx)
            cand_preds_all = cand.sum_predictions(prev_data_arr, True)

            cand_sub_inds = bag_idx



            # (B) score on ALL splits
            deltas, improved = [], 0
            if use_weights:
                # candidate weights: recompute for the “candidate-at-slot” ensemble
                tmp_layer = list(pot_final_layer)
                tmp_layer[slot] = cand
                tmp_inds = list(L3_sub_inds)
                tmp_inds[slot] = cand_sub_inds
                weights_cand = modify_data.compute_oob_weights(tmp_layer, tmp_inds, prev_data_arr)  # len=T
                weights_cand_ext = np.append(weights_cand, 0.0)

                for si, sp in enumerate(val_splits):
                    tmp_val = sp["L3_val"].copy()
                    tmp_val[:, slot] = cand_preds_all[sp["val_idx"]]
                    tmp_full = np.column_stack([tmp_val, sp["y_val"]])  # (N_val, T+1)
                    MV = modify_data.Majority_voting_weighted(tmp_full, weights_cand_ext)
                    score = modify_data.score(MV[:, 0], MV[:, 1])
                    delta = score - base_scores[si]  # weighted vs weighted ✅
                    deltas.append(delta)
                    if delta > 0:
                        improved += 1
            else:
                for si, sp in enumerate(val_splits):
                    tmp_val = sp["L3_val"].copy()
                    tmp_val[:, slot] = cand_preds_all[sp["val_idx"]]
                    tmp_full = np.column_stack([tmp_val, sp["y_val"]])
                    MV = modify_data.Majority_voting(tmp_full)
                    score = modify_data.score(MV[:, 0], MV[:, 1])
                    delta = score - base_scores[si]  # unweighted vs unweighted ✅
                    deltas.append(delta)
                    if delta > 0:
                        improved += 1

            mean_delta = float(np.mean(deltas))

            # keep the most promising even if gains are tiny
            if (improved >= 3 and mean_delta >= -EPS) and (mean_delta > best_mean_delta + 1e-6):
                best_mean_delta = mean_delta
                best_tree = cand
                best_preds_all = cand_preds_all
                best_sub_inds = cand_sub_inds

        # (C) commit if not worse on average (within EPS) and shows at least some split gain
        if best_tree is not None and best_mean_delta >= -EPS:
            pot_final_layer[slot] = best_tree
            pot_final_layer_data[:, slot] = best_preds_all
            accepted_count += 1
            accepted_slots.append(int(slot))

            L3_sub_inds[slot] = best_sub_inds

            # update cached VAL matrices and baselines for subsequent slots
            for si, sp in enumerate(val_splits):
                sp["L3_val"][:, slot] = best_preds_all[sp["val_idx"]]
                tmp_full = np.column_stack([sp["L3_val"], sp["y_val"]])

                if use_weights:
                    base_weights = modify_data.compute_oob_weights(pot_final_layer, L3_sub_inds, prev_data_arr)
                    base_weights_ext = np.append(base_weights, 0.0)
                    MV = modify_data.Majority_voting_weighted(tmp_full, base_weights_ext)
                    base_scores[si] = modify_data.score(MV[:, 0], MV[:, 1])  # overwrite same list
                else:
                    MV = modify_data.Majority_voting(tmp_full)
                    base_scores[si] = modify_data.score(MV[:, 0], MV[:, 1])

    layers.append(pot_final_layer)
    ret_data.append(pot_final_layer_data)

    final_weights = modify_data.compute_oob_weights(pot_final_layer, L3_sub_inds, prev_data_arr)
    if use_weights:
        full_train = np.column_stack([ret_data[-1], y_true_all])
        MV_data = modify_data.Majority_voting_weighted(full_train, np.append(final_weights, 0.0))
    else:
        MV_data = modify_data.Majority_voting(ret_data[-1])
    pot_acc = modify_data.score(MV_data[:, 0], MV_data[:, 1])
    #print_train_score(MV_data, "propagate")

    return layers, ret_data, pot_acc, feature, accepted_count, accepted_slots, L3_sub_inds, final_weights


def train_single(Layers, feature_set, Data, Change_data, include_random_trees, biased_data, biased_classes, distribution, worst_tree, L3_sub_inds, use_weights = True):
    pot_prev_layer = Layers[0].copy()
    prev_layer_data_copy = deepcopy(Change_data[0])
    accepted_count = 0
    accepted_slots = []
    for i in range(len(feature_set)):
        pot_prev_layer_tree = Trees.DecisionTree()
        pot_prev_layer_tree.train(Data, False, None, biased_data[i], biased_classes[i])
        pot_prev_layer_trainpreds = pot_prev_layer_tree.sum_predictions(Data, True)
        pot_prev_layer[feature_set[i]] = pot_prev_layer_tree
        prev_layer_data_copy[:, feature_set[i]] = pot_prev_layer_trainpreds.copy()

    prev_data = np.array(prev_layer_data_copy)
    layers = [pot_prev_layer]
    ret_data = [prev_data]

    # ---------------- L3 retrain with multi-split VAL ----------------
    pot_final_layer      = deepcopy(Layers[-1])       # list of L3 trees
    pot_final_layer_data = deepcopy(Change_data[-1])  # [n_rows, n_trees] preds for ALL train rows

    prev_data_arr = np.asarray(prev_data)             # L2 outputs for L3
    y_true_all    = Data[:, -1]

    # config

    # --- build S stratified validation splits
    # ==== Multi-split validation setup (drop into both functions before the slot loop) ====
    K = 5  # or 4; small is fine
    rng = np.random.RandomState(42)

    # stratified K splits
    val_splits = []
    for _ in range(K):
        val_idx_list = []
        trn_idx_list = []
        for cls in np.unique(y_true_all):
            cls_idx = np.where(y_true_all == cls)[0]
            rng.shuffle(cls_idx)
            n_val = max(1, int(0.20 * len(cls_idx)))
            val_idx_list.append(cls_idx[:n_val])
            trn_idx_list.append(cls_idx[n_val:])
        val_idx = np.concatenate(val_idx_list)
        trn_idx = np.concatenate(trn_idx_list)
        # L3 view on this split
        L3_val = pot_final_layer_data[val_idx, :].copy()
        y_val = y_true_all[val_idx]
        val_splits.append({
            "val_idx": val_idx,
            "trn_idx": trn_idx,
            "L3_val": L3_val,
            "y_val": y_val
        })
    # --- BASELINES ---
    if use_weights:
        base_weights = modify_data.compute_oob_weights(pot_final_layer, L3_sub_inds, prev_data_arr)  # len=T
        base_weights_ext = np.append(base_weights, 0.0)  # pad for label col
        base_scores = []
        for sp in val_splits:
            base_full = np.column_stack([sp["L3_val"], sp["y_val"]])  # (N_val, T+1)
            MV = modify_data.Majority_voting_weighted(base_full, base_weights_ext)
            base_scores.append(modify_data.score(MV[:, 0], MV[:, 1]))
    else:
        base_scores = [modify_data.score(
            modify_data.Majority_voting(np.column_stack([sp["L3_val"], sp["y_val"]]))[:, 0],
            modify_data.Majority_voting(np.column_stack([sp["L3_val"], sp["y_val"]]))[:, 1]
        ) for sp in val_splits]

        # ---- iterate only the requested L3 slots
    for slot in worst_tree if isinstance(worst_tree, list) and isinstance(worst_tree[-1], (list, np.ndarray)) else worst_tree:
        best_mean_delta = float('-inf')
        best_tree = None
        best_preds_all = None
        best_sub_inds = None

        # knobs
        M_CANDIDATES = 50
        BAG_FRAC = 0.63
        EPS = 0  # allow tiny no-worse changes

        for _ in range(M_CANDIDATES):
            # (A) rotate which split we source training indices from to increase diversity
            if _ >= M_CANDIDATES:
                BAG_FRAC = 0.63
            split_idx = np.random.randint(len(val_splits))
            trn_pool = val_splits[split_idx]["trn_idx"]
            bag_sz = max(1, int(len(trn_pool) * BAG_FRAC))
            if _ >= M_CANDIDATES:
                bag_idx = np.random.choice(trn_pool, bag_sz, replace=True)
            else:
                bag_idx = np.random.choice(trn_pool, size=bag_sz, replace=True)

            cand = Trees.DecisionTree()
            cand.train(prev_data_arr[bag_idx], True, bag_idx)
            cand_preds_all = cand.sum_predictions(prev_data_arr, True)


            cand_sub_inds = bag_idx

            # (B) score on ALL splits
            deltas, improved = [], 0
            if use_weights:
                # candidate weights: recompute for the “candidate-at-slot” ensemble
                tmp_layer = list(pot_final_layer)
                tmp_layer[slot] = cand
                tmp_inds = list(L3_sub_inds)
                tmp_inds[slot] = cand_sub_inds
                weights_cand = modify_data.compute_oob_weights(tmp_layer, tmp_inds, prev_data_arr)  # len=T
                weights_cand_ext = np.append(weights_cand, 0.0)

                for si, sp in enumerate(val_splits):
                    tmp_val = sp["L3_val"].copy()
                    tmp_val[:, slot] = cand_preds_all[sp["val_idx"]]
                    tmp_full = np.column_stack([tmp_val, sp["y_val"]])  # (N_val, T+1)
                    MV = modify_data.Majority_voting_weighted(tmp_full, weights_cand_ext)
                    score = modify_data.score(MV[:, 0], MV[:, 1])
                    delta = score - base_scores[si]  # weighted vs weighted ✅
                    deltas.append(delta)
                    if delta > 0:
                        improved += 1
            else:
                for si, sp in enumerate(val_splits):
                    tmp_val = sp["L3_val"].copy()
                    tmp_val[:, slot] = cand_preds_all[sp["val_idx"]]
                    tmp_full = np.column_stack([tmp_val, sp["y_val"]])
                    MV = modify_data.Majority_voting(tmp_full)
                    score = modify_data.score(MV[:, 0], MV[:, 1])
                    delta = score - base_scores[si]  # unweighted vs unweighted ✅
                    deltas.append(delta)
                    if delta > 0:
                        improved += 1

            mean_delta = float(np.mean(deltas))

            # keep the most promising even if gains are tiny
            if (improved >= 1 and mean_delta >= -EPS) and (mean_delta > best_mean_delta + 1e-6):
                best_mean_delta = mean_delta
                best_tree = cand
                best_preds_all = cand_preds_all
                best_sub_inds = cand_sub_inds

        # (C) commit if not worse on average (within EPS) and shows at least some split gain
        if best_tree is not None and best_mean_delta >= -EPS:
            pot_final_layer[slot] = best_tree
            pot_final_layer_data[:, slot] = best_preds_all
            accepted_count += 1
            accepted_slots.append(int(slot))

            L3_sub_inds[slot] = best_sub_inds

            # update cached VAL matrices and baselines for subsequent slots
            for si, sp in enumerate(val_splits):
                sp["L3_val"][:, slot] = best_preds_all[sp["val_idx"]]
                tmp_full = np.column_stack([sp["L3_val"], sp["y_val"]])

                if use_weights:
                    base_weights = modify_data.compute_oob_weights(pot_final_layer, L3_sub_inds, prev_data_arr)
                    base_weights_ext = np.append(base_weights, 0.0)
                    MV = modify_data.Majority_voting_weighted(tmp_full, base_weights_ext)
                    base_scores[si] = modify_data.score(MV[:, 0], MV[:, 1])  # overwrite same list
                else:
                    MV = modify_data.Majority_voting(tmp_full)
                    base_scores[si] = modify_data.score(MV[:, 0], MV[:, 1])

    layers.append(pot_final_layer)
    ret_data.append(pot_final_layer_data)

    final_weights = modify_data.compute_oob_weights(pot_final_layer, L3_sub_inds, prev_data_arr)
    if use_weights:
        full_train = np.column_stack([ret_data[-1], y_true_all])
        MV_data = modify_data.Majority_voting_weighted(full_train, np.append(final_weights, 0.0))
    else:
        MV_data = modify_data.Majority_voting(ret_data[-1])
    pot_acc = modify_data.score(MV_data[:, 0], MV_data[:, 1])
    #print_train_score(MV_data, "single")

    return layers, ret_data, pot_acc, feature_set, accepted_count, accepted_slots, L3_sub_inds, final_weights

def new_random_features(feature_size, samples, n_classes):
    random_features = []
    for i in range(feature_size):
        rand_feat = np.random.randint(n_classes, size=samples) #<---- from 2 to n_classes
        random_features.append(rand_feat.copy())

    random_features = np.array(random_features).transpose()

    return random_features