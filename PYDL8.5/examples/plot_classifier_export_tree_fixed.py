"""
==================================
Default DL85Classifier export tree
==================================

"""
import graphviz
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pydl85 import DL85Classifier


print("######################################################################\n"
      "#                      DL8.5 default classifier                      #\n"
      "######################################################################")

# read the dataset and split into features and targets
dataset = np.genfromtxt("datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DL85Classifier(max_depth=2, min_sup=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# show results
print("Model built in", round(clf.runtime_, 4), "seconds")
print("Confusion Matrix below\n", confusion_matrix(y_test, y_pred))
print("Accuracy on training set =", round(clf.accuracy_, 4))
print("Accuracy on test set =", round(accuracy_score(y_test, y_pred), 4))

# print the tree
print("Serialized json tree:", clf.tree_)

# Try to export graphviz with error handling
try:
    dot = clf.export_graphviz()
    print("Tree visualization (DOT format):")
    print(dot)
    
    # Try to create and render the graph
    try:
        graph = graphviz.Source(dot, format="png")
        print("Graph object created successfully!")
        # Uncomment the next line to save as PNG (requires system graphviz)
        # graph.render("anneal_tree", cleanup=True)
        print("Tree structure exported successfully!")
    except Exception as render_error:
        print(f"Could not render graph (need system graphviz): {render_error}")
        print("But DOT format is available above for manual visualization")
        
except Exception as export_error:
    print(f"Could not export tree visualization: {export_error}")
    print("Tree structure as JSON:", clf.tree_)
    
    # Create a simple text-based tree visualization
    def print_simple_tree(tree_dict, indent=0):
        spacing = "  " * indent
        if isinstance(tree_dict, dict):
            if 'feat' in tree_dict:  # Internal node
                print(f"{spacing}├─ Feature {tree_dict['feat']} <= {tree_dict.get('th', '?')}")
                if 'left' in tree_dict:
                    print_simple_tree(tree_dict['left'], indent+1)
                if 'right' in tree_dict:
                    print_simple_tree(tree_dict['right'], indent+1)
            else:  # Leaf node
                value = tree_dict.get('value', tree_dict.get('name', 'unknown'))
                print(f"{spacing}└─ Predict: {value}")
        else:
            print(f"{spacing}└─ Value: {tree_dict}")
    
    print("\nSimple text-based tree visualization:")
    print_simple_tree(clf.tree_)

print("\n🎉 DL8.5 tree analysis completed!")
