import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification, load_breast_cancer, load_wine


def debug_baseline_comparison():
    """
    Debug why Decision Tree might be beating Random Forest.
    """
    print("🔍 DEBUGGING BASELINE COMPARISON")
    print("=" * 50)

    # Test on a known dataset first
    print("📊 Testing on Breast Cancer dataset (known benchmark)...")

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {np.unique(y)}")

    # Test different configurations
    configurations = [
        {
            'name': 'Original Config',
            'dt_params': {'max_depth': 30, 'random_state': 42},
            'rf_params': {'n_estimators': 100, 'max_depth': 30, 'random_state': 42}
        },
        {
            'name': 'Unlimited Depth',
            'dt_params': {'max_depth': None, 'random_state': 42},
            'rf_params': {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
        },
        {
            'name': 'Default RF Settings',
            'dt_params': {'max_depth': 30, 'random_state': 42},
            'rf_params': {'n_estimators': 100, 'random_state': 42}  # No max_depth limit
        },
        {
            'name': 'More Trees',
            'dt_params': {'max_depth': 30, 'random_state': 42},
            'rf_params': {'n_estimators': 500, 'max_depth': None, 'random_state': 42}
        }
    ]

    print(f"\n🧪 TESTING DIFFERENT CONFIGURATIONS:")
    print(f"{'Config':<20} {'DT_Train':<9} {'DT_Test':<8} {'RF_Train':<9} {'RF_Test':<8} {'RF_Advantage':<12}")
    print("-" * 75)

    for config in configurations:
        # Train Decision Tree
        dt = DecisionTreeClassifier(**config['dt_params'])
        dt.fit(X_train, y_train)

        dt_train_acc = dt.score(X_train, y_train)
        dt_test_acc = dt.score(X_test, y_test)

        # Train Random Forest
        rf = RandomForestClassifier(**config['rf_params'], n_jobs=-1)
        rf.fit(X_train, y_train)

        rf_train_acc = rf.score(X_train, y_train)
        rf_test_acc = rf.score(X_test, y_test)

        rf_advantage = rf_test_acc - dt_test_acc

        print(
            f"{config['name']:<20} {dt_train_acc:<9.3f} {dt_test_acc:<8.3f} {rf_train_acc:<9.3f} {rf_test_acc:<8.3f} {rf_advantage:<+12.3f}")

        # Additional diagnostics for first config
        if config['name'] == 'Original Config':
            print(f"\n🔍 DETAILED DIAGNOSTICS FOR ORIGINAL CONFIG:")
            print(f"   Decision Tree:")
            print(f"     Actual depth: {dt.get_depth()}")
            print(f"     Number of leaves: {dt.get_n_leaves()}")
            print(f"     Overfitting: {dt_train_acc - dt_test_acc:.3f}")

            print(f"   Random Forest:")
            print(f"     Number of estimators: {rf.n_estimators}")
            print(f"     Max depth setting: {rf.max_depth}")
            print(f"     Overfitting: {rf_train_acc - rf_test_acc:.3f}")

            # Check individual tree depths in RF
            tree_depths = [tree.get_depth() for tree in rf.estimators_]
            print(f"     Average tree depth: {np.mean(tree_depths):.1f}")
            print(f"     Max tree depth: {np.max(tree_depths)}")
            print(f"     Min tree depth: {np.min(tree_depths)}")


def test_synthetic_dataset():
    """
    Test on synthetic dataset with known characteristics.
    """
    print(f"\n📊 TESTING ON SYNTHETIC DATASET...")

    # Create synthetic dataset with redundancy
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=8,  # High redundancy
        n_clusters_per_class=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Redundant features: 8/20 (40%)")

    # Test both models
    dt = DecisionTreeClassifier(max_depth=30, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    dt_test_acc = dt.score(X_test, y_test)
    rf_test_acc = rf.score(X_test, y_test)

    print(f"Decision Tree accuracy: {dt_test_acc:.3f}")
    print(f"Random Forest accuracy: {rf_test_acc:.3f}")
    print(f"RF advantage: {rf_test_acc - dt_test_acc:+.3f}")

    if rf_test_acc <= dt_test_acc:
        print("⚠️  WARNING: Random Forest not outperforming Decision Tree on synthetic data!")
    else:
        print("✅ Expected behavior: Random Forest outperforms Decision Tree")


def check_data_loading_issue():
    """
    Check if there's an issue with data loading/preprocessing.
    """
    print(f"\n🔍 CHECKING DATA LOADING ISSUES...")

    # Simulate what might be happening in your data loading
    test_cases = [
        "Normal data",
        "Data with NaN values",
        "Data with identical features",
        "Data with single class",
        "Very small dataset"
    ]

    for case in test_cases:
        print(f"\n🧪 Testing: {case}")

        if case == "Normal data":
            X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        elif case == "Data with NaN values":
            X, y = make_classification(n_samples=200, n_features=10, random_state=42)
            X[0:5, 0:2] = np.nan  # Add some NaN values
        elif case == "Data with identical features":
            X, y = make_classification(n_samples=200, n_features=10, random_state=42)
            X[:, 5] = X[:, 0]  # Make feature 5 identical to feature 0
            X[:, 6] = X[:, 0]  # Make feature 6 identical to feature 0
        elif case == "Data with single class":
            X = np.random.randn(200, 10)
            y = np.zeros(200)  # All same class
        elif case == "Very small dataset":
            X, y = make_classification(n_samples=20, n_features=10, random_state=42)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Check for obvious issues
            if np.isnan(X_train).any():
                print("   ⚠️  Contains NaN values")
                continue

            if len(np.unique(y_train)) < 2:
                print("   ⚠️  Less than 2 classes in training set")
                continue

            if X_train.shape[0] < 10:
                print("   ⚠️  Very small training set")
                continue

            # Train models
            dt = DecisionTreeClassifier(max_depth=30, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42)

            dt.fit(X_train, y_train)
            rf.fit(X_train, y_train)

            dt_acc = dt.score(X_test, y_test)
            rf_acc = rf.score(X_test, y_test)

            print(f"   DT: {dt_acc:.3f}, RF: {rf_acc:.3f}, Advantage: {rf_acc - dt_acc:+.3f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    debug_baseline_comparison()
    test_synthetic_dataset()
    check_data_loading_issue()

    print(f"\n💡 RECOMMENDATIONS:")
    print(f"1. Check if Random Forest is actually using multiple trees")
    print(f"2. Verify max_depth settings aren't constraining RF too much")
    print(f"3. Ensure data isn't corrupted during loading")
    print(f"4. Try removing max_depth limit for Random Forest")
    print(f"5. Check if datasets are too small for RF to show advantage")