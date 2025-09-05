"""
========================================
Default DL85Classifier on COMPAS dataset with custom visualization
========================================
"""
import time
import graphviz
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pydl85 import DL85Classifier, Cache_Type

dataset = np.genfromtxt("datasets/compas.csv", delimiter=',', skip_header=1)
X, y = dataset[:, :-1], dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# read the column names
with open("datasets/compas.csv", 'r') as f:
    col_names = f.readline().strip().split(',')
    col_names = col_names[:-1]

print("######################################################################\n"
      "#                      DL8.5 default classifier                      #\n"
      "######################################################################")

# Try different parameters to get more interesting trees
clf = DL85Classifier(max_depth=5, min_sup=10, cache_type=Cache_Type.Cache_HashCover)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4))

print("Tree structure:", clf.tree_)

def create_custom_dot(tree_dict, feature_names=None, class_names=None):
    """Custom DOT visualization that handles simple trees"""
    dot = ['digraph Tree {']
    dot.append('node [shape=box, style="filled,rounded", color="black", fontname=helvetica] ;')
    dot.append('edge [fontname=helvetica] ;')
    
    def add_node(node, node_id="0", parent_id=None, edge_label=""):
        if isinstance(node, dict):
            if 'feat' in node and 'th' in node:
                # Internal node
                feat_idx = node['feat']
                feat_name = feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"Feature {feat_idx}"
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
                
                # Use class names if provided
                if class_names and isinstance(value, (int, float)) and 0 <= int(value) < len(class_names):
                    class_label = class_names[int(value)]
                else:
                    class_label = str(value)
                
                label = f"Class: {class_label}\\nError: {error:.3f}\\nSamples: {samples}"
                dot.append(f'leaf_{node_id} [label="{label}", fillcolor="#FFE6E6"] ;')
                
                if parent_id is not None:
                    dot.append(f'node_{parent_id} -> leaf_{node_id} [label="{edge_label}"] ;')
        else:
            # Simple value
            class_label = class_names[int(node)] if class_names and 0 <= int(node) < len(class_names) else str(node)
            dot.append(f'leaf_{node_id} [label="Class: {class_label}", fillcolor="#FFE6E6"] ;')
            if parent_id is not None:
                dot.append(f'node_{parent_id} -> leaf_{node_id} [label="{edge_label}"] ;')
    
    add_node(tree_dict)
    dot.append('}')
    return '\n'.join(dot)

# Create custom visualization
try:
    dot_string = create_custom_dot(clf.tree_, feature_names=col_names, class_names=["No Recidivism", "Recidivism"])
    print("\nCustom DOT visualization:")
    print(dot_string)
    
    # Create graphviz object
    graph = graphviz.Source(dot_string)
    print("\n✅ Graphviz visualization created successfully!")
    
    # Create plots directory and save
    import os
    os.makedirs("plots", exist_ok=True)
    
    graph.render('plots/compas_dl85_tree', format='png', cleanup=True)
    print("✅ PNG file saved as 'plots/compas_dl85_tree.png'")
        
except Exception as e:
    print(f"❌ Visualization error: {e}")

print("\n🎉 COMPAS dataset analysis completed!")
print(f"📊 The optimal tree achieves {accuracy_score(y_test, y_pred):.1%} accuracy")
print("📁 Check plots/ directory for the tree visualization")
