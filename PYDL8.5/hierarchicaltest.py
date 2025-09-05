import numpy as np
from collections import Counter, defaultdict
from itertools import product
from sklearn.model_selection import train_test_split
from pydl85 import DL85Classifier


class UniversalHierarchicalRedundancyMetric:
    """
    Universal hierarchical redundancy measurement for any optimal decision tree.

    Works with any tree structure from DL8.5, sklearn, or other tree algorithms.
    Measures discrete instances of identical hierarchical decision patterns.
    """

    def __init__(self, tree_dict):
        """
        Initialize with any tree structure.

        Args:
            tree_dict: Tree structure with nodes containing:
                      - 'feat': feature index for internal nodes
                      - 'left'/'right': child nodes
                      - 'value': class for leaf nodes
        """
        self.tree_dict = tree_dict
        self.subtree_instances = []

    def extract_hierarchical_signature(self, node, max_depth=3):
        """
        Extract hierarchical decision signature from any subtree.

        Creates a structural fingerprint that captures the exact
        parent-child feature relationships in the decision logic.

        Args:
            node: Current tree node
            max_depth: Maximum levels to include in signature

        Returns:
            Tuple representing hierarchical structure or None if not meaningful
        """
        if max_depth <= 0 or not node or 'feat' not in node:
            return None

        root_feature = node['feat']

        # Extract child signatures
        left_signature = None
        right_signature = None

        if 'left' in node and node['left']:
            if 'feat' in node['left']:
                left_signature = self.extract_hierarchical_signature(
                    node['left'], max_depth - 1
                )
            else:
                left_signature = 'LEAF'
        else:
            left_signature = 'NONE'

        if 'right' in node and node['right']:
            if 'feat' in node['right']:
                right_signature = self.extract_hierarchical_signature(
                    node['right'], max_depth - 1
                )
            else:
                right_signature = 'LEAF'
        else:
            right_signature = 'NONE'

        return (root_feature, left_signature, right_signature)

    def is_hierarchical_pattern(self, signature, min_depth=2):
        """
        Check if signature represents meaningful hierarchical decision pattern.

        Args:
            signature: Hierarchical signature tuple
            min_depth: Minimum depth required for pattern

        Returns:
            True if represents meaningful hierarchical structure
        """
        if not signature or not isinstance(signature, tuple) or len(signature) != 3:
            return False

        root_feature, left_sig, right_sig = signature

        # Must have at least one child with decision structure
        has_decision_child = (
                (left_sig and left_sig not in ['LEAF', 'NONE'] and isinstance(left_sig, tuple)) or
                (right_sig and right_sig not in ['LEAF', 'NONE'] and isinstance(right_sig, tuple))
        )

        return has_decision_child

    def extract_all_hierarchical_patterns(self, node=None, location_path="root", visited=None):
        """
        Extract all hierarchical patterns from the entire tree.

        Args:
            node: Current node (starts with root)
            location_path: String path to current location
            visited: Set of visited node IDs to prevent double-counting
        """
        if node is None:
            node = self.tree_dict
            visited = set()

        if not node or 'feat' not in node:
            return

        # Prevent analyzing the same node multiple times
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        # Extract hierarchical patterns of different depths from this location
        for depth in range(2, 5):  # Check 2-4 level hierarchies
            signature = self.extract_hierarchical_signature(node, depth)

            if signature and self.is_hierarchical_pattern(signature):
                self.subtree_instances.append({
                    'signature': signature,
                    'location': location_path,
                    'root_feature': node['feat'],
                    'depth': depth,
                    'node_id': node_id
                })

        # Recursively analyze child nodes
        if 'left' in node and node['left']:
            self.extract_all_hierarchical_patterns(
                node['left'], f"{location_path}.L", visited
            )

        if 'right' in node and node['right']:
            self.extract_all_hierarchical_patterns(
                node['right'], f"{location_path}.R", visited
            )

    def find_redundant_hierarchical_patterns(self):
        """
        Identify hierarchical patterns that appear in multiple distinct locations.

        Returns:
            Dictionary mapping signatures to lists of instances
        """
        # Extract all patterns
        self.extract_all_hierarchical_patterns()

        # Group by signature, ensuring unique node instances
        signature_groups = defaultdict(list)
        processed_combinations = set()

        for instance in self.subtree_instances:
            signature = instance['signature']
            node_id = instance['node_id']
            depth = instance['depth']

            # Create unique key to avoid counting same node at different depths
            unique_key = (signature, node_id, depth)

            if unique_key not in processed_combinations:
                processed_combinations.add(unique_key)
                signature_groups[signature].append(instance)

        # Keep only patterns with 2+ instances (redundant)
        redundant_patterns = {
            sig: instances for sig, instances in signature_groups.items()
            if len(instances) >= 2
        }

        return redundant_patterns

    def signature_to_readable_pattern(self, signature):
        """
        Convert hierarchical signature to human-readable pattern string.

        Args:
            signature: Hierarchical signature tuple

        Returns:
            String like "7-8-9" representing the feature sequence
        """

        def extract_feature_sequence(sig):
            """Recursively extract features from signature."""
            if not sig or sig in ['LEAF', 'NONE']:
                return []

            if isinstance(sig, tuple) and len(sig) >= 1:
                features = [sig[0]]  # Root feature

                # Add features from children
                if len(sig) >= 2:
                    features.extend(extract_feature_sequence(sig[1]))
                if len(sig) >= 3:
                    features.extend(extract_feature_sequence(sig[2]))

                return features

            return []

        feature_sequence = extract_feature_sequence(signature)
        unique_features = sorted(list(set(feature_sequence)))
        return '-'.join(map(str, unique_features))

    def calculate_hierarchical_redundancy_score(self):
        """
        Calculate comprehensive hierarchical redundancy score.

        Returns:
            Tuple of (redundancy_score, redundant_patterns_dict)
        """
        redundant_patterns = self.find_redundant_hierarchical_patterns()

        if not self.subtree_instances:
            return 0.0, {}

        # Count total instances of redundant patterns
        total_redundant_instances = sum(
            len(instances) for instances in redundant_patterns.values()
        )

        # Calculate total unique subtree instances analyzed
        unique_instances = len(set(
            (inst['signature'], inst['node_id'])
            for inst in self.subtree_instances
        ))

        # Redundancy score
        redundancy_score = total_redundant_instances / max(unique_instances, 1)

        return redundancy_score, redundant_patterns

    def analyze_hierarchical_redundancy(self, verbose=True):
        """
        Complete hierarchical redundancy analysis for any decision tree.

        Args:
            verbose: Whether to print detailed analysis

        Returns:
            Dictionary with analysis results
        """
        if verbose:
            print("UNIVERSAL HIERARCHICAL REDUNDANCY ANALYSIS")
            print("=" * 50)

        # Calculate redundancy
        redundancy_score, redundant_patterns = self.calculate_hierarchical_redundancy_score()

        # Classify redundancy level
        if redundancy_score >= 0.6:
            level = "HIGH"
            deep_forest_prediction = "STRONG advantage expected"
        elif redundancy_score >= 0.3:
            level = "MODERATE"
            deep_forest_prediction = "Some advantage expected"
        else:
            level = "LOW"
            deep_forest_prediction = "Limited advantage expected"

        if verbose:
            print(f"Hierarchical Pattern Analysis:")
            print(f"  Total hierarchical subtrees analyzed: {len(self.subtree_instances)}")
            print(f"  Redundant pattern types found: {len(redundant_patterns)}")
            total_redundant = sum(len(instances) for instances in redundant_patterns.values())
            print(f"  Total redundant instances: {total_redundant}")
            print(f"  Hierarchical redundancy score: {redundancy_score:.3f}")
            print(f"  Redundancy level: {level}")
            print(f"  Deep Forest prediction: {deep_forest_prediction}")

            # Show specific redundant patterns
            if redundant_patterns:
                print(f"\nRedundant Hierarchical Patterns:")
                sorted_patterns = sorted(
                    redundant_patterns.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )

                for signature, instances in sorted_patterns:
                    count = len(instances)
                    pattern_name = self.signature_to_readable_pattern(signature)
                    print(f"  {count}x {pattern_name}")

                    # Show first few locations
                    if count <= 8:
                        locations = [inst['location'] for inst in instances[:5]]
                        if len(locations) < len(instances):
                            locations.append("...")
                        print(f"    Locations: {', '.join(locations)}")
            else:
                print(f"\nNo redundant hierarchical patterns found.")

        return {
            'redundancy_score': redundancy_score,
            'redundancy_level': level,
            'deep_forest_prediction': deep_forest_prediction,
            'redundant_patterns': redundant_patterns,
            'total_patterns': len(redundant_patterns),
            'suitable_for_deep_forest': level in ['HIGH', 'MODERATE']
        }


def measure_tree_hierarchical_redundancy(tree_dict, verbose=True):
    """
    Universal function to measure hierarchical redundancy in any decision tree.

    Args:
        tree_dict: Any decision tree structure with 'feat', 'left', 'right' nodes
        verbose: Whether to print analysis details

    Returns:
        Dictionary with redundancy analysis results

    Usage:
        # For DL8.5 trees:
        results = measure_tree_hierarchical_redundancy(clf.tree_)

        # For sklearn trees (requires conversion):
        # results = measure_tree_hierarchical_redundancy(sklearn_tree_to_dict(clf))

        # Check if suitable for Deep Forest research:
        if results['suitable_for_deep_forest']:
            print("Dataset suitable for testing Deep Forest hierarchical advantages")
    """
    analyzer = UniversalHierarchicalRedundancyMetric(tree_dict)
    results = analyzer.analyze_hierarchical_redundancy(verbose=verbose)

    if verbose:
        print(f"\nDEEP FOREST RESEARCH SUITABILITY:")
        if results['suitable_for_deep_forest']:
            print(f"  ✓ Dataset suitable for Deep Forest testing")
            print(f"  ✓ Expected performance advantage: {results['deep_forest_prediction']}")
        else:
            print(f"  ✗ Dataset not ideal for Deep Forest hierarchical validation")
            print(f"  ✗ {results['deep_forest_prediction']}")

    return results


def analyze_your_boolean_tree():
    """
    Analyze your specific boolean function tree for hierarchical redundancy.
    """
    print("ANALYZING YOUR BOOLEAN FUNCTION DECISION TREE")
    print("=" * 60)

    # Your exact boolean function
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

    # Train optimal tree with your settings
    clf = DL85Classifier(max_depth=8, min_sup=1, max_error=0)
    clf.fit(X_train, y_train)

    print(f"\nTree Performance:")
    print(f"  Training accuracy: {clf.score(X_train, y_train):.3f}")
    print(f"  Test accuracy: {clf.score(X_test, y_test):.3f}")

    # Analyze hierarchical redundancy in the trained tree
    print(f"\n" + "=" * 60)
    results = measure_tree_hierarchical_redundancy(clf.tree_, verbose=True)

    return clf, results


# Run the analysis
if __name__ == "__main__":
    clf, results = analyze_your_boolean_tree()