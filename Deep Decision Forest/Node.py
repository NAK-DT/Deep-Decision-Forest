class Node:
    def __init__(self, data = None, feature = None, threshold = None, information_gain = None, left = None, right = None, value = None, parent = None, accuracy = 0, data_index = None, index = 0, depth = 0, NoDP = 0, alt_feat = None, train_pred_data = None, train_pred_index = None):
        self.index = index
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
        self.depth = depth
        self.NoDP = NoDP
        self.alt_feat = alt_feat
        self.train_pred_data = train_pred_data
        self.train_pred_index = train_pred_index