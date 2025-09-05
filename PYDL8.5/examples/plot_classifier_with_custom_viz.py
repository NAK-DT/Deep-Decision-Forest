"""
DL85Classifier with custom tree visualization
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

from pydl85 import DL85Classifier

print("######################################################################\n"
      "#                      DL8.5 with custom visualization               #\n"
      "######################################################################")

# Load dataset
dataset = np.genfromtxt("datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Try with slightly deeper tree to get more interesting visualization
clf = DL85Classifier(max_depth=3, min_sup=5)  # Increased depth and min_sup
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Model built in", round(clf.runtime_, 4), "seconds")
print("Confusion Matrix below\n", confusion_matrix(y_test, y_pred))
print("Accuracy on training set =", round(clf.accuracy_, 4))
print("Accuracy on test set =", round(accuracy_score(y_test, y_pred), 4))
print("Tree JSON:", clf.tree_)

def create_custom_dot(tree_dict, feature_names=None):
    """Custom DOT visualization that handles simple trees"""
    dot = ['digraph Tree {']
    dot.append('node [shape=box, style="filled,rounded", color="black", fontname=helvetica] ;')
    dot.append('edge [fontname=helvetica] ;')
    
    def add_node(node, node_id="0", parent_id=None, edge_label=""):
        if isinstance(node, dict):
            if 'feat' in node and 'th' in node:
                # Internal node
                feat_name = f"Feature {node['feat']}" if not feature_names else feature_names[node['feat']]
                label = f"{feat_name}\\n<= {node['th']}"
                dot.append(f'node_{node_id} [label="{label}", fillcolor="#E6F3FF"] ;')
                
                if parent_id is not None:
                    dot.append(f'node_{parent_id} -> node_{node_id} [label="{edge_label}"] ;')
                
                # Add children
                if 'left' in node:
                    add_node(node['left'], f"{node_id}_L", node_id, "True")
                if 'right' in node:
                    add_node(node['right'], f"{node_id}_R", node_id, "False")
            else:
                # Leaf node
                value = node.get('value', node.get('name', 'unknown'))
                error = node.get('error', 0.0)
                samples = node.get('size', '?')
                
                label = f"Class: {value}\\nError: {error:.3f}\\nSamples: {samples}"
                dot.append(f'leaf_{node_id} [label="{label}", fillcolor="#FFE6E6"] ;')
                
                if parent_id is not None:
                    dot.append(f'node_{parent_id} -> leaf_{node_id} [label="{edge_label}"] ;')
        else:
            # Simple value
            dot.append(f'leaf_{node_id} [label="Class: {node}", fillcolor="#FFE6E6"] ;')
            if parent_id is not None:
                dot.append(f'node_{parent_id} -> leaf_{node_id} [label="{edge_label}"] ;')
    
    add_node(tree_dict)
    dot.append('}')
    return '\n'.join(dot)

# Create custom visualization
try:
    dot_string = create_custom_dot(clf.tree_)
    print("\nCustom DOT visualization:")
    print(dot_string)
    
    # Create graphviz object
    graph = graphviz.Source(dot_string)
    print("\n✅ Graphviz visualization created successfully!")
    
    # Try to render (comment out if no system graphviz)
    try:
        graph.render('custom_dl85_tree', format='png', cleanup=True)
        print("✅ PNG file saved as 'custom_dl85_tree.png'")
    except Exception as e:
        print(f"⚠️  Could not save PNG (need system graphviz): {e}")
        print("But you can copy the DOT string above to an online graphviz viewer!")
        
except Exception as e:
    print(f"❌ Visualization error: {e}")

print("\n🎉 Custom visualization completed!")
