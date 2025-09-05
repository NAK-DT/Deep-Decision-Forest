import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
import warnings

warnings.filterwarnings('ignore')


class MultiDatasetAnalyzer:
    """
    Analyze redundancy across all your datasets and test Deep Forest performance.
    """

    def __init__(self, datasets_folder="datasets"):
        self.datasets_folder = datasets_folder
        self.results = {}

    def load_dataset(self, filename):
        """
        Load and preprocess a single dataset.
        Handles different formats and encoding issues.
        """
        filepath = os.path.join(self.datasets_folder, filename)
        dataset_name = filename.replace('.txt', '').replace('.csv', '')

        try:
            # Try different delimiters
            for delimiter in [' ', ',', '\t', ';']:
                try:
                    if filename.endswith('.csv'):
                        data = pd.read_csv(filepath, delimiter=delimiter)
                    else:
                        data = pd.read_csv(filepath, delimiter=delimiter, header=None)

                    # Check if data loaded properly
                    if data.shape[1] > 1:
                        break
                except:
                    continue

            # Convert to numpy array
            if isinstance(data, pd.DataFrame):
                data = data.values

            # Handle different target column positions
            # Try target as last column first
            if data.shape[1] > 1:
                X = data[:, :-1].astype(float)
                y = data[:, -1]
            else:
                raise ValueError("Dataset has only one column")

            # Encode categorical targets
            if y.dtype == 'object' or len(np.unique(y)) < 20:
                le = LabelEncoder()
                y = le.fit_transform(y)

            print(f"✅ Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
            return X, y, dataset_name

        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            return None, None, None

    def calculate_redundancy_score(self, X, y=None):
        """
        Calculate redundancy score for any dataset (reusing our previous method).
        """
        try:
            n_features = X.shape[1]

            # 1. Correlation redundancy
            corr_matrix = np.corrcoef(X.T)
            high_corr_count = 0
            total_pairs = 0

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if not np.isnan(corr_matrix[i, j]):
                        if abs(corr_matrix[i, j]) > 0.7:
                            high_corr_count += 1
                        total_pairs += 1

            corr_score = high_corr_count / max(total_pairs, 1)

            # 2. PCA redundancy
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA()
            pca.fit(X_scaled)

            cumvar = np.cumsum(pca.explained_variance_ratio_)
            components_needed = np.argmax(cumvar >= 0.95) + 1
            pca_score = max(0, 1 - (components_needed / n_features))

            # Combined score
            final_score = 0.5 * corr_score + 0.5 * pca_score

            return final_score, corr_score, pca_score

        except Exception as e:
            print(f"⚠️  Error calculating redundancy: {e}")
            return 0.0, 0.0, 0.0

    def analyze_single_dataset(self, filename):
        """
        Complete analysis of a single dataset.
        """
        X, y, dataset_name = self.load_dataset(filename)

        if X is None:
            return None

        # Calculate redundancy
        redundancy_score, corr_score, pca_score = self.calculate_redundancy_score(X, y)

        # Determine redundancy level
        if redundancy_score >= 0.6:
            level = "HIGH"
            prediction = "Strong Deep Forest advantage expected"
        elif redundancy_score >= 0.3:
            level = "MODERATE"
            prediction = "Some Deep Forest advantage expected"
        else:
            level = "LOW"
            prediction = "Limited Deep Forest advantage expected"

        # Store results
        result = {
            'dataset_name': dataset_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'redundancy_score': redundancy_score,
            'correlation_component': corr_score,
            'pca_component': pca_score,
            'redundancy_level': level,
            'prediction': prediction,
            'X': X,
            'y': y
        }

        self.results[dataset_name] = result

        print(f"\n📊 {dataset_name.upper()}")
        print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
        print(f"   Redundancy Score: {redundancy_score:.3f}")
        print(f"   Level: {level}")
        print(f"   Prediction: {prediction}")

        return result

    def analyze_all_datasets(self):
        """
        Analyze all datasets in the folder.
        """
        print("🔍 ANALYZING ALL DATASETS")
        print("=" * 50)

        # Get all dataset files
        dataset_files = [f for f in os.listdir(self.datasets_folder)
                         if f.endswith(('.txt', '.csv'))]

        print(f"Found {len(dataset_files)} datasets to analyze...")

        successful_analyses = 0
        for filename in sorted(dataset_files):
            result = self.analyze_single_dataset(filename)
            if result is not None:
                successful_analyses += 1

        print(f"\n✅ Successfully analyzed {successful_analyses}/{len(dataset_files)} datasets")

        return self.results

    def generate_research_summary(self):
        """
        Generate summary for your research paper.
        """
        if not self.results:
            print("No results to summarize. Run analyze_all_datasets() first.")
            return

        print(f"\n📋 RESEARCH SUMMARY - DATASET REDUNDANCY ANALYSIS")
        print("=" * 60)

        # Create summary table
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Dataset': name,
                'Samples': result['n_samples'],
                'Features': result['n_features'],
                'Classes': result['n_classes'],
                'Redundancy': f"{result['redundancy_score']:.3f}",
                'Level': result['redundancy_level'],
                'Deep Forest Expected': result['redundancy_level'] in ['HIGH', 'MODERATE']
            })

        # Sort by redundancy score
        summary_data.sort(key=lambda x: float(x['Redundancy']), reverse=True)

        # Print table
        print(
            f"{'Dataset':<20} {'Samples':<8} {'Features':<9} {'Classes':<8} {'Redundancy':<11} {'Level':<9} {'DF Advantage':<12}")
        print("-" * 85)

        high_redundancy_count = 0
        moderate_redundancy_count = 0
        low_redundancy_count = 0

        for row in summary_data:
            df_advantage = "✅ Yes" if row['Deep Forest Expected'] else "❌ No"
            print(
                f"{row['Dataset']:<20} {row['Samples']:<8} {row['Features']:<9} {row['Classes']:<8} {row['Redundancy']:<11} {row['Level']:<9} {df_advantage:<12}")

            if row['Level'] == 'HIGH':
                high_redundancy_count += 1
            elif row['Level'] == 'MODERATE':
                moderate_redundancy_count += 1
            else:
                low_redundancy_count += 1

        # Summary statistics
        print(f"\n📊 DATASET DISTRIBUTION:")
        print(f"   HIGH redundancy (≥0.6): {high_redundancy_count} datasets")
        print(f"   MODERATE redundancy (0.3-0.6): {moderate_redundancy_count} datasets")
        print(f"   LOW redundancy (<0.3): {low_redundancy_count} datasets")

        total_datasets = len(summary_data)
        expected_advantage = high_redundancy_count + moderate_redundancy_count

        print(f"\n🎯 RESEARCH PREDICTIONS:")
        print(
            f"   Deep Forest advantage expected: {expected_advantage}/{total_datasets} datasets ({expected_advantage / total_datasets:.1%})")
        print(
            f"   Strong advantage expected: {high_redundancy_count}/{total_datasets} datasets ({high_redundancy_count / total_datasets:.1%})")

        return summary_data

    def get_datasets_by_redundancy_level(self, level):
        """
        Get datasets of a specific redundancy level for targeted experiments.

        Args:
            level: 'HIGH', 'MODERATE', or 'LOW'
        """
        filtered_datasets = {}
        for name, result in self.results.items():
            if result['redundancy_level'] == level:
                filtered_datasets[name] = result

        return filtered_datasets

    def run_baseline_experiments(self, test_size=0.2, random_state=42):
        """
        Run Random Forest and Decision Tree baselines on all datasets.

        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        print(f"\n🏃 RUNNING BASELINE EXPERIMENTS")
        print("=" * 50)
        print(f"Configuration:")
        print(f"  • Decision Tree: max_depth=30")
        print(f"  • Random Forest: n_estimators=100, max_depth=30")
        print(f"  • Test size: {test_size:.1%}")

        baseline_results = {}

        for dataset_name, dataset_info in self.results.items():
            print(f"\n🔬 Testing {dataset_name}...")

            X, y = dataset_info['X'], dataset_info['y']

            # Data validation
            if np.isnan(X).any():
                print(f"   ⚠️  Warning: Dataset contains NaN values, skipping...")
                continue

            if len(np.unique(y)) < 2:
                print(f"   ⚠️  Warning: Dataset has less than 2 classes, skipping...")
                continue

            if X.shape[0] < 20:
                print(f"   ⚠️  Warning: Dataset too small ({X.shape[0]} samples), skipping...")
                continue

            # Train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if len(np.unique(y)) > 1 else None
                )
            except ValueError:
                # Fallback if stratification fails
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

            # Initialize models with better configurations
            dt = DecisionTreeClassifier(max_depth=30, random_state=random_state)
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            # Note: Removed max_depth limit for RF to allow optimal depth finding

            dataset_results = {
                'dataset_info': dataset_info,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }

            # Train and evaluate Decision Tree
            start_time = time.time()
            dt.fit(X_train, y_train)
            dt_train_time = time.time() - start_time

            dt_train_pred = dt.predict(X_train)
            dt_test_pred = dt.predict(X_test)

            dt_train_acc = accuracy_score(y_train, dt_train_pred)
            dt_test_acc = accuracy_score(y_test, dt_test_pred)
            dt_train_f1 = f1_score(y_train, dt_train_pred, average='weighted')
            dt_test_f1 = f1_score(y_test, dt_test_pred, average='weighted')

            dataset_results['decision_tree'] = {
                'train_accuracy': dt_train_acc,
                'test_accuracy': dt_test_acc,
                'train_f1': dt_train_f1,
                'test_f1': dt_test_f1,
                'training_time': dt_train_time,
                'tree_depth': dt.get_depth(),
                'n_leaves': dt.get_n_leaves(),
                'overfitting': dt_train_acc - dt_test_acc
            }

            # Train and evaluate Random Forest
            start_time = time.time()
            rf.fit(X_train, y_train)
            rf_train_time = time.time() - start_time

            rf_train_pred = rf.predict(X_train)
            rf_test_pred = rf.predict(X_test)

            rf_train_acc = accuracy_score(y_train, rf_train_pred)
            rf_test_acc = accuracy_score(y_test, rf_test_pred)
            rf_train_f1 = f1_score(y_train, rf_train_pred, average='weighted')
            rf_test_f1 = f1_score(y_test, rf_test_pred, average='weighted')

            dataset_results['random_forest'] = {
                'train_accuracy': rf_train_acc,
                'test_accuracy': rf_test_acc,
                'train_f1': rf_train_f1,
                'test_f1': rf_test_f1,
                'training_time': rf_train_time,
                'overfitting': rf_train_acc - rf_test_acc
            }

            baseline_results[dataset_name] = dataset_results

            # Print results for this dataset with verification
            print(f"   Decision Tree: {dt_test_acc:.3f} accuracy ({dt_train_acc:.3f} train) - Depth: {dt.get_depth()}")
            print(f"   Random Forest: {rf_test_acc:.3f} accuracy ({rf_train_acc:.3f} train)")
            rf_advantage = rf_test_acc - dt_test_acc
            print(f"   RF advantage: {rf_advantage:+.3f}")

            # Sanity check - RF should usually be better
            if rf_advantage < -0.05:  # RF significantly worse than DT
                print(f"   ⚠️  WARNING: RF performing much worse than DT - check data quality")

        self.baseline_results = baseline_results
        return baseline_results

    def generate_baseline_summary(self):
        """
        Generate comprehensive summary of baseline results.
        """
        if not hasattr(self, 'baseline_results'):
            print("No baseline results found. Run run_baseline_experiments() first.")
            return

        print(f"\n📊 BASELINE RESULTS SUMMARY")
        print("=" * 80)

        # Create summary table
        summary_data = []
        for dataset_name, results in self.baseline_results.items():
            dataset_info = results['dataset_info']
            dt_results = results['decision_tree']
            rf_results = results['random_forest']

            summary_data.append({
                'Dataset': dataset_name,
                'Redundancy': f"{dataset_info['redundancy_score']:.3f}",
                'Level': dataset_info['redundancy_level'],
                'DT_Acc': f"{dt_results['test_accuracy']:.3f}",
                'RF_Acc': f"{rf_results['test_accuracy']:.3f}",
                'RF_Advantage': f"{rf_results['test_accuracy'] - dt_results['test_accuracy']:+.3f}",
                'DT_Overfit': f"{dt_results['overfitting']:.3f}",
                'RF_Overfit': f"{rf_results['overfitting']:.3f}",
                'Features': dataset_info['n_features'],
                'Samples': dataset_info['n_samples']
            })

        # Sort by redundancy score
        summary_data.sort(key=lambda x: float(x['Redundancy']), reverse=True)

        # Print table header
        print(
            f"{'Dataset':<18} {'Red.':<5} {'Level':<4} {'DT_Acc':<6} {'RF_Acc':<6} {'RF_Adv':<6} {'DT_Over':<7} {'RF_Over':<7} {'Feat':<4} {'Samp':<6}")
        print("-" * 80)

        # Print results
        for row in summary_data:
            print(
                f"{row['Dataset']:<18} {row['Redundancy']:<5} {row['Level']:<4} {row['DT_Acc']:<6} {row['RF_Acc']:<6} {row['RF_Advantage']:<6} {row['DT_Overfit']:<7} {row['RF_Overfit']:<7} {row['Features']:<4} {row['Samples']:<6}")

        # Analyze patterns
        print(f"\n🔍 PATTERN ANALYSIS:")

        # Group by redundancy level
        high_red = [row for row in summary_data if row['Level'] == 'HIGH']
        mod_red = [row for row in summary_data if row['Level'] == 'MODERATE']
        low_red = [row for row in summary_data if row['Level'] == 'LOW']

        def avg_rf_advantage(group):
            if not group:
                return 0
            return np.mean([float(row['RF_Advantage']) for row in group])

        high_avg_adv = avg_rf_advantage(high_red)
        mod_avg_adv = avg_rf_advantage(mod_red)
        low_avg_adv = avg_rf_advantage(low_red)

        print(f"   Average RF advantage by redundancy level:")
        print(f"     HIGH redundancy:     {high_avg_adv:+.3f} (n={len(high_red)})")
        print(f"     MODERATE redundancy: {mod_avg_adv:+.3f} (n={len(mod_red)})")
        print(f"     LOW redundancy:      {low_avg_adv:+.3f} (n={len(low_red)})")

        # Correlation analysis
        redundancy_scores = [float(row['Redundancy']) for row in summary_data]
        rf_advantages = [float(row['RF_Advantage']) for row in summary_data]

        correlation = np.corrcoef(redundancy_scores, rf_advantages)[0, 1]
        print(f"\n📈 CORRELATION ANALYSIS:")
        print(f"   Correlation between redundancy and RF advantage: {correlation:.3f}")

        if correlation > 0.3:
            print(f"   ✅ POSITIVE correlation: Higher redundancy → Better RF performance")
        elif correlation < -0.3:
            print(f"   ❌ NEGATIVE correlation: Higher redundancy → Worse RF performance")
        else:
            print(f"   ⚠️  WEAK correlation: No clear relationship")

        # Deep Forest predictions
        print(f"\n🎯 DEEP FOREST PREDICTIONS:")
        print(f"   For your Deep Forest to validate the hypothesis, it should:")

        for row in summary_data:
            if row['Level'] == 'HIGH':
                target_acc = float(row['RF_Acc']) + 0.02  # Should beat RF by 2%+
                print(f"     {row['Dataset']:<18}: Beat {row['RF_Acc']} (target: >{target_acc:.3f})")

        return summary_data

    def save_results_to_csv(self, filename="dataset_analysis_results.csv"):
        """
        Save all results to CSV for further analysis.
        """
        if not hasattr(self, 'baseline_results'):
            print("No baseline results to save. Run experiments first.")
            return

        # Combine redundancy and baseline results
        export_data = []
        for dataset_name, baseline_results in self.baseline_results.items():
            dataset_info = baseline_results['dataset_info']
            dt_results = baseline_results['decision_tree']
            rf_results = baseline_results['random_forest']

            export_data.append({
                'dataset': dataset_name,
                'n_samples': dataset_info['n_samples'],
                'n_features': dataset_info['n_features'],
                'n_classes': dataset_info['n_classes'],
                'redundancy_score': dataset_info['redundancy_score'],
                'redundancy_level': dataset_info['redundancy_level'],
                'dt_test_accuracy': dt_results['test_accuracy'],
                'dt_train_accuracy': dt_results['train_accuracy'],
                'dt_overfitting': dt_results['overfitting'],
                'dt_depth': dt_results['tree_depth'],
                'dt_leaves': dt_results['n_leaves'],
                'rf_test_accuracy': rf_results['test_accuracy'],
                'rf_train_accuracy': rf_results['train_accuracy'],
                'rf_overfitting': rf_results['overfitting'],
                'rf_advantage_over_dt': rf_results['test_accuracy'] - dt_results['test_accuracy'],
                'dt_training_time': dt_results['training_time'],
                'rf_training_time': rf_results['training_time']
            })

        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")

        return df


# =============================================================================
# USAGE FOR YOUR RESEARCH
# =============================================================================

def main():
    """
    Main function to analyze all your datasets.
    """
    # Initialize analyzer
    analyzer = MultiDatasetAnalyzer("datasets")  # Make sure folder path is correct

    # Analyze all datasets
    results = analyzer.analyze_all_datasets()

    # Generate research summary
    summary = analyzer.generate_research_summary()

    # Get high-redundancy datasets for priority testing
    high_redundancy_datasets = analyzer.get_datasets_by_redundancy_level('HIGH')
    print(f"\n🎯 HIGH REDUNDANCY DATASETS FOR PRIORITY TESTING:")
    for name in high_redundancy_datasets.keys():
        print(f"   • {name}")

    # Example: Prepare data for specific experiment
    if 'mushroom' in results:
        experiment_data = analyzer.prepare_experiment_data('mushroom')
        print(f"\n🧪 Example: Mushroom dataset prepared for experiments")
        print(f"   Training samples: {len(experiment_data['X_train'])}")
        print(f"   Test samples: {len(experiment_data['X_test'])}")

    return analyzer, results, summary


# AUTO-RUN: Analyze redundancy and run baseline experiments
print("🚀 STARTING COMPREHENSIVE DATASET ANALYSIS")
print("=" * 60)

# Initialize analyzer with your datasets folder
analyzer = MultiDatasetAnalyzer("datasets")

# Step 1: Analyze redundancy of all datasets
print("📊 STEP 1: ANALYZING REDUNDANCY...")
results = analyzer.analyze_all_datasets()

# Step 2: Run baseline experiments (Decision Tree + Random Forest)
print("🏃 STEP 2: RUNNING BASELINE EXPERIMENTS...")
baseline_results = analyzer.run_baseline_experiments()

# Step 3: Generate comprehensive analysis
print("📋 STEP 3: GENERATING RESEARCH SUMMARY...")
summary = analyzer.generate_baseline_summary()

# Step 4: Save results for further analysis
analyzer.save_results_to_csv("dataset_analysis_results.csv")

# Additional insights for your research
high_redundancy = analyzer.get_datasets_by_redundancy_level('HIGH')
moderate_redundancy = analyzer.get_datasets_by_redundancy_level('MODERATE')

print(f"\n🎯 NEXT STEPS FOR YOUR DEEP FOREST:")
print(f"1. Test your Deep Forest on these HIGH redundancy datasets:")
for name in high_redundancy.keys():
    rf_acc = analyzer.baseline_results[name]['random_forest']['test_accuracy']
    target = rf_acc + 0.02
    print(f"   • {name:<20}: Beat RF accuracy of {rf_acc:.3f} (target: >{target:.3f})")

print(f"\n2. Your Deep Forest should show STRONGEST advantage on HIGH redundancy datasets")
print(f"3. Use LOW redundancy datasets as negative controls (where advantage should be minimal)")
print(f"4. Results saved to 'dataset_analysis_results.csv' for statistical analysis")

print(f"\n✅ ANALYSIS COMPLETE! Ready for Deep Forest experiments.")

if __name__ == "__main__":
    pass  # Everything runs automatically above