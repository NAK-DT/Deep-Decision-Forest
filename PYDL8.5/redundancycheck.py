import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
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

    def prepare_experiment_data(self, dataset_name, test_size=0.2):
        """
        Prepare train/test splits for a specific dataset for your experiments.
        """
        if dataset_name not in self.results:
            print(f"Dataset {dataset_name} not found. Available: {list(self.results.keys())}")
            return None

        result = self.results[dataset_name]
        X, y = result['X'], result['y']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'dataset_info': result
        }


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


# =============================================================================
# AUTO-RUN ON YOUR DATASETS FOLDER
# =============================================================================

print("🚀 STARTING AUTOMATIC ANALYSIS OF YOUR DATASETS")
print("=" * 60)

# Initialize analyzer with your datasets folder
analyzer = MultiDatasetAnalyzer("datasets")

# Analyze all datasets automatically
results = analyzer.analyze_all_datasets()

# Generate comprehensive research summary
summary = analyzer.generate_research_summary()

# Get datasets categorized by redundancy level
high_redundancy = analyzer.get_datasets_by_redundancy_level('HIGH')
moderate_redundancy = analyzer.get_datasets_by_redundancy_level('MODERATE')
low_redundancy = analyzer.get_datasets_by_redundancy_level('LOW')

print(f"\n🎯 EXPERIMENTAL PRIORITIES FOR YOUR RESEARCH:")
print(f"\n🔴 HIGH REDUNDANCY - Test Deep Forest here FIRST:")
for name in high_redundancy.keys():
    score = high_redundancy[name]['redundancy_score']
    print(f"   • {name:<20} (score: {score:.3f}) - STRONG advantage expected")

print(f"\n🟡 MODERATE REDUNDANCY - Test these SECOND:")
for name in moderate_redundancy.keys():
    score = moderate_redundancy[name]['redundancy_score']
    print(f"   • {name:<20} (score: {score:.3f}) - Some advantage expected")

print(f"\n🟢 LOW REDUNDANCY - Test these for COMPARISON:")
for name in low_redundancy.keys():
    score = low_redundancy[name]['redundancy_score']
    print(f"   • {name:<20} (score: {score:.3f}) - Limited advantage expected")

# Additional insights
total_datasets = len(results)
if total_datasets > 0:
    avg_redundancy = np.mean([r['redundancy_score'] for r in results.values()])
    high_count = len(high_redundancy)

    print(f"\n📈 OVERALL INSIGHTS:")
    print(f"   Total datasets analyzed: {total_datasets}")
    print(f"   Average redundancy score: {avg_redundancy:.3f}")
    print(f"   Datasets with expected Deep Forest advantage: {high_count + len(moderate_redundancy)}/{total_datasets}")
    print(f"   Your hypothesis should be STRONGEST on: {list(high_redundancy.keys())}")

    if high_count >= 3:
        print(f"\n✅ EXCELLENT! You have {high_count} high-redundancy datasets to validate your hypothesis")
    elif high_count >= 1:
        print(f"\n⚠️  MODERATE: You have {high_count} high-redundancy dataset(s) - may need more for strong validation")
    else:
        print(f"\n❌ WARNING: No high-redundancy datasets found - your hypothesis may be hard to validate")

print(f"\n🔬 NEXT STEPS:")
print(f"1. Run Deep Forest experiments on HIGH redundancy datasets first")
print(f"2. Compare against baselines (Random Forest, XGBoost, etc.)")
print(f"3. Correlate performance improvements with redundancy scores")
print(f"4. Use LOW redundancy datasets as negative controls")
print(f"\n💾 Results stored in 'analyzer.results' for further analysis")

if __name__ == "__main__":
    pass  # Everything runs automatically above