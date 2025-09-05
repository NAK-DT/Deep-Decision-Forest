import numpy as np
from sklearn.model_selection import train_test_split
from pydl85 import DL85Classifier
import graphviz
from itertools import product
from collections import Counter, defaultdict
import json


class TreeRedundancyAnalyzer:
    """
    Analyze redundancy patterns in optimal decision trees.
    """

    def __init__(self, tree_dict):
        self.tree_dict = tree_dict
        self.patterns = []
        self.subtree_patterns = []
        self.feature_paths = []

    def extract_all_patterns(self):
        """
        Extract all possible patterns from the tree structure.
        """
        # Extract different types of patterns
        self.extract_feature_sequences()
        self.extract_subtree_structures()
        self.extract_decision_paths()

        return {
            'feature_sequences': self.get_feature_sequence_redundancy(),
            'subtree_structures': self.get_subtree_redundancy(),
            'decision_paths': self.get_path_redundancy(),
            'overall_redundancy_score': self.calculate_overall_redundancy()
        }

    def extract_feature_sequences(self, node=None, current_path=None, depth=0):
        """
        Extract sequences of features used in decision paths.
        """
        if node is None:
            node = self.tree_dict
            current_path = []

        if 'feat' in node:  # Internal node
            feature = node['feat']
            new_path = current_path + [feature]

            # Store paths of different lengths
            for length in range(2, len(new_path) + 1):
                if length <= len(new_path):
                    sequence = tuple(new_path[-length:])
                    self.patterns.append(('feature_sequence', sequence, depth))

            # Recursively process children
            if 'left' in node:
                self.extract_feature_sequences(node['left'], new_path, depth + 1)
            if 'right' in node:
                self.extract_feature_sequences(node['right'], new_path, depth + 1)

    def extract_subtree_structures(self, node=None, depth=0):
        """
        Extract structural patterns of subtrees.
        """
        if node is None:
            node = self.tree_dict

        if 'feat' in node:  # Internal node
            # Create structure signature
            structure = self.get_subtree_signature(node)
            self.subtree_patterns.append(('subtree_structure', structure, depth))

            # Process children
            if 'left' in node:
                self.extract_subtree_structures(node['left'], depth + 1)
            if 'right' in node:
                self.extract_subtree_structures(node['right'], depth + 1)

    def get_subtree_signature(self, node, max_depth=3):
        """
        Create a signature for a subtree structure.
        """
        if max_depth == 0 or 'feat' not in node:
            return 'leaf'

        feature = node.get('feat', 'leaf')
        left_sig = 'none'
        right_sig = 'none'

        if 'left' in node:
            left_sig = self.get_subtree_signature(node['left'], max_depth - 1)
        if 'right' in node:
            right_sig = self.get_subtree_signature(node['right'], max_depth - 1)

        return f"f{feature}({left_sig},{right_sig})"

    def extract_decision_paths(self, node=None, current_path=None):
        """
        Extract complete decision paths from root to leaves.
        """
        if node is None:
            node = self.tree_dict
            current_path = []

        if 'feat' in node:  # Internal node
            feature = node['feat']

            # Process left child (feature = 0)
            if 'left' in node:
                left_path = current_path + [(feature, 0)]
                self.extract_decision_paths(node['left'], left_path)

            # Process right child (feature = 1)
            if 'right' in node:
                right_path = current_path + [(feature, 1)]
                self.extract_decision_paths(node['right'], right_path)
        else:
            # Leaf node - store complete path
            class_value = node.get('value', '?')
            self.feature_paths.append((tuple(current_path), class_value))

    def get_feature_sequence_redundancy(self):
        """
        Analyze redundancy in feature sequences.
        """
        sequences = [pattern[1] for pattern in self.patterns if pattern[0] == 'feature_sequence']
        sequence_counts = Counter(sequences)

        redundant_sequences = {seq: count for seq, count in sequence_counts.items() if count > 1}

        total_sequences = len(sequences)
        redundant_count = sum(count for count in redundant_sequences.values())

        return {
            'redundant_sequences': dict(redundant_sequences),
            'total_sequences': total_sequences,
            'redundancy_ratio': redundant_count / max(total_sequences, 1),
            'most_common': sequence_counts.most_common(5)
        }

    def get_subtree_redundancy(self):
        """
        Analyze redundancy in subtree structures.
        """
        structures = [pattern[1] for pattern in self.subtree_patterns if pattern[0] == 'subtree_structure']
        structure_counts = Counter(structures)

        redundant_structures = {struct: count for struct, count in structure_counts.items() if count > 1}

        total_structures = len(structures)
        redundant_count = sum(count for count in redundant_structures.values())

        return {
            'redundant_structures': dict(redundant_structures),
            'total_structures': total_structures,
            'redundancy_ratio': redundant_count / max(total_structures, 1),
            'most_common': structure_counts.most_common(5)
        }

    def get_path_redundancy(self):
        """
        Analyze redundancy in decision paths.
        """
        # Group paths by class
        paths_by_class = defaultdict(list)
        for path, class_val in self.feature_paths:
            paths_by_class[class_val].append(path)

        # Find common subpaths
        all_subpaths = []
        for path, _ in self.feature_paths:
            for length in range(2, len(path) + 1):
                for start in range(len(path) - length + 1):
                    subpath = path[start:start + length]
                    all_subpaths.append(subpath)

        subpath_counts = Counter(all_subpaths)
        redundant_subpaths = {path: count for path, count in subpath_counts.items() if count > 1}

        return {
            'paths_by_class': {k: len(v) for k, v in paths_by_class.items()},
            'redundant_subpaths': dict(redundant_subpaths),
            'total_paths': len(self.feature_paths),
            'redundancy_ratio': len(redundant_subpaths) / max(len(all_subpaths), 1)
        }

    def calculate_overall_redundancy(self):
        """
        Calculate overall redundancy score for the tree.
        """
        seq_redundancy = self.get_feature_sequence_redundancy()['redundancy_ratio']
        struct_redundancy = self.get_subtree_redundancy()['redundancy_ratio']
        path_redundancy = self.get_path_redundancy()['redundancy_ratio']

        # Weighted average
        overall_score = (0.4 * seq_redundancy + 0.4 * struct_redundancy + 0.2 * path_redundancy)

        return {
            'sequence_component': seq_redundancy,
            'structure_component': struct_redundancy,
            'path_component': path_redundancy,
            'overall_score': overall_score,
            'redundancy_level': 'HIGH' if overall_score > 0.5 else 'MODERATE' if overall_score > 0.2 else 'LOW'
        }

    def print_redundancy_report(self):
        """
        Print comprehensive redundancy analysis report.
        """
        results = self.extract_all_patterns()

        print("🌳 DECISION TREE REDUNDANCY ANALYSIS")
        print("=" * 50)

        # Feature sequence redundancy
        seq_results = results['feature_sequences']
        print(f"\n📋 FEATURE SEQUENCE REDUNDANCY:")
        print(f"   Total sequences found: {seq_results['total_sequences']}")
        print(f"   Redundancy ratio: {seq_results['redundancy_ratio']:.3f}")
        print(f"   Most common sequences:")
        for seq, count in seq_results['most_common']:
            if count > 1:
                print(f"     {seq}: appears {count} times")

        # Subtree structure redundancy
        struct_results = results['subtree_structures']
        print(f"\n🌲 SUBTREE STRUCTURE REDUNDANCY:")
        print(f"   Total structures found: {struct_results['total_structures']}")
        print(f"   Redundancy ratio: {struct_results['redundancy_ratio']:.3f}")
        print(f"   Most common structures:")
        for struct, count in struct_results['most_common']:
            if count > 1:
                print(f"     {struct}: appears {count} times")

        # Decision path redundancy
        path_results = results['decision_paths']
        print(f"\n🛤️  DECISION PATH REDUNDANCY:")
        print(f"   Total decision paths: {path_results['total_paths']}")
        print(f"   Redundancy ratio: {path_results['redundancy_ratio']:.3f}")
        print(f"   Paths by class: {path_results['paths_by_class']}")

        # Overall redundancy
        overall = results['overall_redundancy_score']
        print(f"\n🎯 OVERALL REDUNDANCY SCORE:")
        print(f"   Sequence component: {overall['sequence_component']:.3f}")
        print(f"   Structure component: {overall['structure_component']:.3f}")
        print(f"   Path component: {overall['path_component']:.3f}")
        print(f"   Overall score: {overall['overall_score']:.3f}")
        print(f"   Redundancy level: {overall['redundancy_level']}")

        return results

    def visualize_redundant_patterns(self, save_path="redundant_patterns"):
        """
        Create visualization highlighting redundant patterns.
        """
        results = self.extract_all_patterns()

        # Get redundant sequences
        redundant_seqs = results['feature_sequences']['redundant_sequences']

        # Create enhanced dot representation
        dot = ['digraph TreeRedundancy {']
        dot.append('node [shape=box, style="filled,rounded", color="black", fontname=helvetica];')
        dot.append('edge [fontname=helvetica];')

        counter = {'id': 0}

        def new_id():
            counter['id'] += 1
            return counter['id']

        def add_node_with_redundancy(node, current_id=0, path=[]):
            if 'feat' in node:
                feature = node['feat']
                new_path = path + [feature]

                # Check if current path contains redundant sequences
                is_redundant = False
                for length in range(2, len(new_path) + 1):
                    if length <= len(new_path):
                        sequence = tuple(new_path[-length:])
                        if sequence in redundant_seqs:
                            is_redundant = True
                            break

                # Color based on redundancy
                color = "lightcoral" if is_redundant else "lightblue"
                label = f'Feature {feature}'
                if is_redundant:
                    label += '\\n(Redundant)'

                dot.append(f'{current_id} [label="{label}", fillcolor="{color}"];')

                if 'left' in node:
                    left_id = new_id()
                    add_node_with_redundancy(node['left'], left_id, new_path)
                    dot.append(f'{current_id} -> {left_id} [label="0"];')

                if 'right' in node:
                    right_id = new_id()
                    add_node_with_redundancy(node['right'], right_id, new_path)
                    dot.append(f'{current_id} -> {right_id} [label="1"];')
            else:
                value = node.get('value', '?')
                error = node.get('error', 0)
                label = f'Class {value}\\nError: {error}'
                dot.append(f'{current_id} [label="{label}", fillcolor="lightgreen"];')

        add_node_with_redundancy(self.tree_dict)
        dot.append('}')

        # Create and save visualization
        dot_string = '\n'.join(dot)
        graph = graphviz.Source(dot_string)

        try:
            graph.render(save_path, format='png', cleanup=True)
            print(f"🎨 Redundancy visualization saved as {save_path}.png")
            print("🔴 Red nodes indicate redundant patterns")
        except Exception as e:
            print(f"Visualization issue: {e}")

        return dot_string


# Function to analyze your boolean function tree
def analyze_boolean_tree_redundancy():
    """
    Analyze redundancy in the boolean function decision tree.
    """

    # Generate boolean function data
    def boolean_function(row):
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = row
        expr1 = f1 and (f2 or (f3 and f4))
        expr2 = (not f1) and (f5 or (f6 and f7))
        expr3 = f8 and (f9 or (f10 and f3))
        expr4 = (not f8) and (f2 or (f5 and f9))
        return int(expr1 or expr2 or expr3 or expr4)

    n_features = 10
    X_full = np.array(list(product([0, 1], repeat=n_features)))
    y_full = np.array([boolean_function(row) for row in X_full])

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    # Train optimal tree
    clf = DL85Classifier(max_depth=8, min_sup=1, max_error=0)
    clf.fit(X_train, y_train)

    # Analyze redundancy
    analyzer = TreeRedundancyAnalyzer(clf.tree_)
    results = analyzer.print_redundancy_report()

    # Create visualizations
    analyzer.visualize_redundant_patterns("boolean_tree_redundancy")

    # Also create original tree for comparison
    def create_original_tree_viz(tree_dict):
        dot = ['digraph Tree {']
        dot.append('node [shape=box, style="filled,rounded", color="black", fontname=helvetica];')
        dot.append('edge [fontname=helvetica];')

        counter = {'id': 0}

        def new_id():
            counter['id'] += 1
            return counter['id']

        def add_node(node, current_id=0):
            if 'feat' in node:
                feat = node['feat']
                label = f'Feature {feat}'
                dot.append(f'{current_id} [label="{label}", fillcolor="lightblue"];')

                if 'left' in node:
                    left_id = new_id()
                    add_node(node['left'], left_id)
                    dot.append(f'{current_id} -> {left_id} [label="0"];')

                if 'right' in node:
                    right_id = new_id()
                    add_node(node['right'], right_id)
                    dot.append(f'{current_id} -> {right_id} [label="1"];')
            else:
                value = node.get('value', '?')
                error = node.get('error', 0)
                label = f'Class {value}\\nError: {error}'
                dot.append(f'{current_id} [label="{label}", fillcolor="lightgreen"];')

        add_node(tree_dict)
        dot.append('}')
        return '\n'.join(dot)

    # Save original tree
    original_dot = create_original_tree_viz(clf.tree_)
    original_graph = graphviz.Source(original_dot)
    try:
        original_graph.render('boolean_tree_original', format='png', cleanup=True)
        print("🌳 Original tree saved as boolean_tree_original.png")
    except Exception as e:
        print(f"Original tree visualization issue: {e}")

    print(f"\n📊 TREE PERFORMANCE:")
    print(f"   Training accuracy: {clf.score(X_train, y_train):.3f}")
    print(f"   Test accuracy: {clf.score(X_test, y_test):.3f}")

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = analyze_boolean_tree_redundancy()