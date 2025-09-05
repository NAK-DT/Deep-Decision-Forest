
import numpy as np

def cross_val(tree, Data):
    #kf = KFold(n_splits=5,shuffle=False)
    #splits = list(kf.split(Data))
    #print(splits)
    data_list = []
    for split in range(5):
        data_list.append(Data[int(split*0.2*len(Data)):int(((split*0.2)+0.2)*len(Data)),:])
    split_results = []
    for i in range(len(data_list)):
        test = data_list[i]
        train = data_list[:i] + data_list[i+1:]
        con_train = np.array(train[0])
        for j in range(1,len(train)):
            con_train = np.vstack((con_train, train[j]))
        tree.train(con_train)
        split_results.append(tree.sum_predictions(np.array(test[:,:-1])))

    final_preds = split_results[0]
    #print(len(split_results))
    for i in range(1,len(split_results)):
        final_preds = np.hstack((final_preds, split_results[i]))
        #print(final_preds.shape)
    #print(final_preds)
    return final_preds


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

def summarise_layer2_data(Biased_Holder, length):
    print("Start summarising layer2 data")
    Node_data_base, Node_class_base, Node_feature_base = Biased_Holder[0]
    for i in range(len(Node_data_base)):
        print(len(Node_data_base[i])==len(Node_class_base[i]))
    print("Set base")

    for i in range(1,length):
        Temp_Node_data, Temp_Node_classes, Temp_Node_features = Biased_Holder[i]
        for feature in Temp_Node_features:
            if np.isin(feature, Node_feature_base):
                print("Feature exists")
                feature_index_base = Node_feature_base.index(feature)
                feature_index_temp = Temp_Node_features.index(feature)
                base_list = Node_data_base[feature_index_base]
                base_class_list = Node_class_base[feature_index_base]
                temp_data_list = Temp_Node_data[feature_index_temp]
                temp_class_list = Temp_Node_classes[feature_index_temp]
                #check = np.array(temp_data_list)
                #print("temp data list shape: ", temp_data_list)
                for sample in temp_data_list:
                    #print(sample)
                    #print("The sample: ", sample)
                    temp_class_index = temp_data_list.index(sample)
                    #print("Index from temp class index: ", temp_class_index)
                    #print(base_list)
                    if np.isin(sample, base_list):
                        print("Sample exists")
                        #print("node class list: ", Node_class_base)
                        #print("temp_node_classes: ", Temp_Node_classes)
                        class_index = base_list.index(sample)
                        #print("Shape of initial Node class base: ", Node_class_base)
                        #print("indices: ", feature_index_base,"  ",class_index, " ", feature_index_temp, " ", temp_class_index)
                        #print("Node class base: ", len(Node_class_base), " node class base fib: ", len(Node_class_base[feature_index_base]))
                        #print("Node class temp: ", len(Temp_Node_classes), " node class temp fit: ", len(Temp_Node_classes[feature_index_temp]))
                        #print(len(Temp_Node_classes[feature_index_temp]) == len(temp_data_list))
                        print("Type of Node_class_base[feature_index_base]:", type(Node_class_base[feature_index_base]))
                        print("Before appending, Node_class_base[feature_index_base]:",Node_class_base[feature_index_base])
                        Node_class_base[feature_index_base][class_index].extend(Temp_Node_classes[feature_index_temp][temp_class_index])
                    else:
                        print("Sample not exists")
                        #print("indices: ", feature_index_base, "  ", feature_index_temp, " ", temp_class_index)
                        #print("Node class base: ", len(Node_class_base), " node class base fib: ", len(Node_class_base[feature_index_base]))
                        #print("Node class temp: ", len(Temp_Node_classes), " node class temp fit: ", len(Temp_Node_classes[feature_index_temp]))
                        #print(len(Temp_Node_classes[feature_index_temp])==len(temp_data_list))
                        Node_data_base[feature_index_base].append(sample)
                        #print("Type of Node_class_base[feature_index_base]:", type(Node_class_base[feature_index_base]))
                        #print("Before appending, Node_class_base[feature_index_base]:",Node_class_base[feature_index_base])
                        id = len(Node_class_base[feature_index_base])
                        Node_class_base[feature_index_base].append(Temp_Node_classes[feature_index_temp][temp_class_index])
                        id2 = len(Node_class_base[feature_index_base])
                        print(id2-id)

            else:
                print("Feature not exists")
                Node_feature_base.append(feature)
                Node_data_base.append(Temp_Node_data)
                Node_class_base.append(Temp_Node_classes)

    return Node_data_base, Node_class_base, Node_feature_base

def summarise_layer2_data(Biased_Holder, length):
    print("Start summarising layer2 data")
    Node_data_base, Node_class_base, Node_feature_base = Biased_Holder[0]
    for i in range(len(Node_data_base)):
        print(len(Node_data_base[i])==len(Node_class_base[i]))
    print("Set base")

    for i in range(1,length):
        Temp_Node_data, Temp_Node_classes, Temp_Node_features = Biased_Holder[i]
        for feature in Temp_Node_features:
            if np.isin(feature, Node_feature_base):
                print("Feature exists")
                feature_index_base = Node_feature_base.index(feature)
                feature_index_temp = Temp_Node_features.index(feature)
                base_list = Node_data_base[feature_index_base]
                base_class_list = Node_class_base[feature_index_base]
                temp_data_list = Temp_Node_data[feature_index_temp]
                temp_class_list = Temp_Node_classes[feature_index_temp]
                #check = np.array(temp_data_list)
                #print("temp data list shape: ", temp_data_list)
                for sample in temp_data_list:
                    #print(sample)
                    #print("The sample: ", sample)
                    temp_class_index = temp_data_list.index(sample)
                    #print("Index from temp class index: ", temp_class_index)
                    #print(base_list)
                    if np.isin(sample, base_list):
                        print("Sample exists")
                        #print("node class list: ", Node_class_base)
                        #print("temp_node_classes: ", Temp_Node_classes)
                        class_index = base_list.index(sample)
                        #print("Shape of initial Node class base: ", Node_class_base)
                        #print("indices: ", feature_index_base,"  ",class_index, " ", feature_index_temp, " ", temp_class_index)
                        #print("Node class base: ", len(Node_class_base), " node class base fib: ", len(Node_class_base[feature_index_base]))
                        #print("Node class temp: ", len(Temp_Node_classes), " node class temp fit: ", len(Temp_Node_classes[feature_index_temp]))
                        #print(len(Temp_Node_classes[feature_index_temp]) == len(temp_data_list))
                        print("Type of Node_class_base[feature_index_base]:", type(Node_class_base[feature_index_base]))
                        print("Before appending, Node_class_base[feature_index_base]:",Node_class_base[feature_index_base])
                        Node_class_base[feature_index_base][class_index].extend(Temp_Node_classes[feature_index_temp][temp_class_index])
                    else:
                        print("Sample not exists")
                        #print("indices: ", feature_index_base, "  ", feature_index_temp, " ", temp_class_index)
                        #print("Node class base: ", len(Node_class_base), " node class base fib: ", len(Node_class_base[feature_index_base]))
                        #print("Node class temp: ", len(Temp_Node_classes), " node class temp fit: ", len(Temp_Node_classes[feature_index_temp]))
                        #print(len(Temp_Node_classes[feature_index_temp])==len(temp_data_list))
                        Node_data_base[feature_index_base].append(sample)
                        #print("Type of Node_class_base[feature_index_base]:", type(Node_class_base[feature_index_base]))
                        #print("Before appending, Node_class_base[feature_index_base]:",Node_class_base[feature_index_base])
                        id = len(Node_class_base[feature_index_base])
                        Node_class_base[feature_index_base].append(Temp_Node_classes[feature_index_temp][temp_class_index])
                        id2 = len(Node_class_base[feature_index_base])
                        print(id2-id)

            else:
                print("Feature not exists")
                Node_feature_base.append(feature)
                Node_data_base.append(Temp_Node_data)
                Node_class_base.append(Temp_Node_classes)

    return Node_data_base, Node_class_base, Node_feature_base

def summarise_layer2_data(Biased_Holder, length):
    print("Start summarising layer2 data")
    Node_data_base, Node_class_base, Node_feature_base = Biased_Holder[0]
    for i in range(len(Node_data_base)):
        print(len(Node_data_base[i])==len(Node_class_base[i]))
    print("Set base")

    for i in range(1,length):
        Temp_Node_data, Temp_Node_classes, Temp_Node_features = Biased_Holder[i]
        for feature in Temp_Node_features:
            if np.isin(feature, Node_feature_base):
                print("Feature exists")
                feature_index_base = Node_feature_base.index(feature)
                feature_index_temp = Temp_Node_features.index(feature)
                base_list = Node_data_base[feature_index_base]
                base_class_list = Node_class_base[feature_index_base]
                temp_data_list = Temp_Node_data[feature_index_temp]
                temp_class_list = Temp_Node_classes[feature_index_temp]
                #check = np.array(temp_data_list)
                #print("temp data list shape: ", temp_data_list)
                for sample in temp_data_list:
                    #print(sample)
                    #print("The sample: ", sample)
                    temp_class_index = temp_data_list.index(sample)
                    #print("Index from temp class index: ", temp_class_index)
                    #print(base_list)
                    if np.isin(sample, base_list):
                        print("Sample exists")
                        #print("node class list: ", Node_class_base)
                        #print("temp_node_classes: ", Temp_Node_classes)
                        class_index = base_list.index(sample)
                        #print("Shape of initial Node class base: ", Node_class_base)
                        #print("indices: ", feature_index_base,"  ",class_index, " ", feature_index_temp, " ", temp_class_index)
                        #print("Node class base: ", len(Node_class_base), " node class base fib: ", len(Node_class_base[feature_index_base]))
                        #print("Node class temp: ", len(Temp_Node_classes), " node class temp fit: ", len(Temp_Node_classes[feature_index_temp]))
                        #print(len(Temp_Node_classes[feature_index_temp]) == len(temp_data_list))
                        print("Type of Node_class_base[feature_index_base]:", type(Node_class_base[feature_index_base]))
                        print("Before appending, Node_class_base[feature_index_base]:",Node_class_base[feature_index_base])
                        Node_class_base[feature_index_base][class_index].extend(Temp_Node_classes[feature_index_temp][temp_class_index])
                    else:
                        print("Sample not exists")
                        #print("indices: ", feature_index_base, "  ", feature_index_temp, " ", temp_class_index)
                        #print("Node class base: ", len(Node_class_base), " node class base fib: ", len(Node_class_base[feature_index_base]))
                        #print("Node class temp: ", len(Temp_Node_classes), " node class temp fit: ", len(Temp_Node_classes[feature_index_temp]))
                        #print(len(Temp_Node_classes[feature_index_temp])==len(temp_data_list))
                        Node_data_base[feature_index_base].append(sample)
                        #print("Type of Node_class_base[feature_index_base]:", type(Node_class_base[feature_index_base]))
                        #print("Before appending, Node_class_base[feature_index_base]:",Node_class_base[feature_index_base])
                        id = len(Node_class_base[feature_index_base])
                        Node_class_base[feature_index_base].append(Temp_Node_classes[feature_index_temp][temp_class_index])
                        id2 = len(Node_class_base[feature_index_base])
                        print(id2-id)

            else:
                print("Feature not exists")
                Node_feature_base.append(feature)
                Node_data_base.append(Temp_Node_data)
                Node_class_base.append(Temp_Node_classes)

    return Node_data_base, Node_class_base, Node_feature_base