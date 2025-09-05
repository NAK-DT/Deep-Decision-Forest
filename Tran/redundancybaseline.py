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


class ImprovedDatasetAnalyzer:
    """
    Enhanced analyzer that handles categorical data properly.
    """

    def __init__(self, datasets_folder="raw_datasets"):
        self.datasets_folder = datasets_folder
        self.results = {}

    def load_dataset(self, filename):
        """
        Load and preprocess datasets with proper handling for different formats.
        """
        filepath = os.path.join(self.datasets_folder, filename)
        dataset_name = filename.replace('.data', '').replace('.txt', '').replace('.csv', '')

        try:
            # Special handling for known datasets
            if 'breast-cancer' in filename.lower() or 'breastcancer' in filename.lower():
                # Load with header for breast cancer dataset
                data = pd.read_csv(filepath, header=0)
                # Drop ID column if present
                if 'id' in data.columns.str.lower():
                    data = data.drop(data.columns[0], axis=1)

                # Target is diagnosis column
                if 'diagnosis' in data.columns:
                    y_raw = data['diagnosis']
                    X_raw = data.drop('diagnosis', axis=1)
                else:
                    # Fallback if no diagnosis column
                    X_raw = data.iloc[:, :-1]
                    y_raw = data.iloc[:, -1]

                print(f"Raw data shape: {data.shape}")
                print(f"Sample of first few values: {X_raw.iloc[0].tolist()[:5]}")
                print(f"Target sample values: {y_raw.unique()}")

            elif 'mushroom' in filename.lower():
                # Load mushroom dataset with header
                data = pd.read_csv(filepath, header=0)

                # Find the class/target column
                if 'class' in data.columns:
                    y_raw = data['class']
                    X_raw = data.drop('class', axis=1)
                else:
                    # Use first column as target if no 'class' column
                    y_raw = data.iloc[:, 0]
                    X_raw = data.iloc[:, 1:]

                print(f"Raw data shape: {data.shape}")
                print(f"Sample of first few values: {X_raw.iloc[0].tolist()[:5]}")
                print(f"Target sample values: {y_raw.unique()}")

            else:
                # Try different delimiters for other datasets
                for delimiter in [',', ' ', '\t', ';']:
                    try:
                        data = pd.read_csv(filepath, delimiter=delimiter, header=None)
                        if data.shape[1] > 1:
                            break
                    except:
                        continue

                print(f"Raw data shape: {data.shape}")
                print(f"Sample of first few values: {data.iloc[0].tolist()[:5]}")

                # Handle target variable (usually last column)
                if data.shape[1] > 1:
                    X_raw = data.iloc[:, :-1]
                    y_raw = data.iloc[:, -1]
                else:
                    raise ValueError("Dataset has only one column")

            # Encode features properly
            X_encoded = self.encode_features_properly(X_raw)

            # Encode target variable
            if y_raw.dtype == 'object' or not pd.api.types.is_numeric_dtype(y_raw):
                le_target = LabelEncoder()
                y = le_target.fit_transform(y_raw)
                print(f"Target classes: {le_target.classes_}")
            else:
                y = y_raw.values

            print(
                f"✅ Loaded {dataset_name}: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features, {len(np.unique(y))} classes")
            return X_encoded, y, dataset_name

        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            return None, None, None

    def encode_features_properly(self, X_raw):
        """
        Encode features properly using one-hot encoding for categorical, keep numeric as numeric.
        """
        X_processed = X_raw.copy()

        # Separate categorical and numeric columns
        categorical_cols = []
        numeric_cols = []

        for col in X_processed.columns:
            if pd.api.types.is_numeric_dtype(X_processed[col]):
                numeric_cols.append(col)
            else:
                # Check if it's actually numeric but stored as string
                try:
                    pd.to_numeric(X_processed[col], errors='raise')
                    numeric_cols.append(col)
                    X_processed[col] = pd.to_numeric(X_processed[col])
                    print(f"  Converted string to numeric: column {col}")
                except (ValueError, TypeError):
                    categorical_cols.append(col)

        # One-hot encode categorical columns
        if categorical_cols:
            print(f"  One-hot encoding {len(categorical_cols)} categorical columns")
            categorical_data = X_processed[categorical_cols]

            # Use pandas get_dummies for one-hot encoding
            categorical_encoded = pd.get_dummies(categorical_data, drop_first=False)

            # Combine with numeric columns
            if numeric_cols:
                numeric_data = X_processed[numeric_cols]
                final_data = pd.concat([numeric_data, categorical_encoded], axis=1)
            else:
                final_data = categorical_encoded

            print(f"  Final feature count after encoding: {final_data.shape[1]}")
        else:
            print(f"  All {len(numeric_cols)} columns are numeric")
            final_data = X_processed

        return final_data.values.astype(float)

    def calculate_redundancy_score(self, X, y=None):
        """
        Calculate redundancy score with better handling for categorical data.
        """
        try:
            n_features = X.shape[1]

            # 1. Correlation redundancy (with handling for categorical data)
            try:
                corr_matrix = np.corrcoef(X.T)
                # Handle NaN correlations (can happen with constant features)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                high_corr_count = 0
                total_pairs = 0

                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        correlation = abs(corr_matrix[i, j])
                        if correlation > 0.7:
                            high_corr_count += 1
                        total_pairs += 1

                corr_score = high_corr_count / max(total_pairs, 1)
            except:
                corr_score = 0.0
                print("  Warning: Could not calculate correlations")

            # 2. PCA redundancy
            try:
                # Standardize for PCA (important for mixed data types)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA()
                pca.fit(X_scaled)

                cumvar = np.cumsum(pca.explained_variance_ratio_)
                components_needed = np.argmax(cumvar >= 0.95) + 1
                pca_score = max(0, 1 - (components_needed / n_features))

                print(f"  PCA: {components_needed}/{n_features} components for 95% variance")
            except:
                pca_score = 0.0
                print("  Warning: Could not perform PCA")

            # 3. Feature uniqueness redundancy (new for categorical data)
            try:
                uniqueness_scores = []
                for i in range(n_features):
                    unique_ratio = len(np.unique(X[:, i])) / X.shape[0]
                    uniqueness_scores.append(unique_ratio)

                # Low uniqueness indicates potential redundancy
                avg_uniqueness = np.mean(uniqueness_scores)
                uniqueness_redundancy = max(0, 1 - avg_uniqueness * 2)  # Scale appropriately

                print(f"  Avg feature uniqueness: {avg_uniqueness:.3f}")
            except:
                uniqueness_redundancy = 0.0

            # Combined score with three components
            final_score = 0.4 * corr_score + 0.4 * pca_score + 0.2 * uniqueness_redundancy

            return final_score, corr_score, pca_score

        except Exception as e:
            print(f"⚠️  Error calculating redundancy: {e}")
            return 0.0, 0.0, 0.0

    def analyze_single_dataset(self, filename):
        """
        Complete analysis of a single dataset with better error handling.
        """
        print(f"\n🔍 Analyzing {filename}...")
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

        print(f"📊 {dataset_name.upper()}")
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
                         if f.endswith(('.txt', '.csv', '.data'))]

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
        Generate summary for research.
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

        high_count = moderate_count = low_count = 0

        for row in summary_data:
            df_advantage = "✅ Yes" if row['Deep Forest Expected'] else "❌ No"
            print(
                f"{row['Dataset']:<20} {row['Samples']:<8} {row['Features']:<9} {row['Classes']:<8} {row['Redundancy']:<11} {row['Level']:<9} {df_advantage:<12}")

            if row['Level'] == 'HIGH':
                high_count += 1
            elif row['Level'] == 'MODERATE':
                moderate_count += 1
            else:
                low_count += 1

        print(f"\n📊 DATASET DISTRIBUTION:")
        print(f"   HIGH redundancy (≥0.6): {high_count} datasets")
        print(f"   MODERATE redundancy (0.3-0.6): {moderate_count} datasets")
        print(f"   LOW redundancy (<0.3): {low_count} datasets")

        return summary_data

    def run_baseline_experiments(self, test_size=0.2, random_state=42):
        """
        Run Decision Tree and Random Forest baselines on all loaded datasets.
        """
        print(f"\n🏃 RUNNING BASELINE EXPERIMENTS")
        print("=" * 50)
        print(f"Configuration:")
        print(f"  • Decision Tree: max_depth=30")
        print(f"  • Random Forest: n_estimators=100, no max_depth limit")
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
                # For multiclass, use stratify if possible
                if len(np.unique(y)) > 2:
                    # Check if stratification is possible
                    min_class_count = min([np.sum(y == cls) for cls in np.unique(y)])
                    if min_class_count >= 2:  # Need at least 2 samples per class for split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
            except ValueError:
                # Fallback if stratification fails
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

            # Initialize models
            dt = DecisionTreeClassifier(max_depth=30, random_state=random_state)
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

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

            # Handle multiclass F1 scoring
            f1_average = 'binary' if len(np.unique(y)) == 2 else 'weighted'
            dt_train_f1 = f1_score(y_train, dt_train_pred, average=f1_average)
            dt_test_f1 = f1_score(y_test, dt_test_pred, average=f1_average)

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
            rf_train_f1 = f1_score(y_train, rf_train_pred, average=f1_average)
            rf_test_f1 = f1_score(y_test, rf_test_pred, average=f1_average)

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
                'Samples': dataset_info['n_samples'],
                'Classes': dataset_info['n_classes']
            })

        # Sort by redundancy score
        summary_data.sort(key=lambda x: float(x['Redundancy']), reverse=True)

        # Print table header
        print(
            f"{'Dataset':<15} {'Red.':<5} {'Level':<4} {'DT_Acc':<6} {'RF_Acc':<6} {'RF_Adv':<6} {'DT_Over':<7} {'RF_Over':<7} {'Feat':<4} {'Samp':<6} {'Cls':<3}")
        print("-" * 85)

        # Print results
        for row in summary_data:
            print(
                f"{row['Dataset']:<15} {row['Redundancy']:<5} {row['Level']:<4} {row['DT_Acc']:<6} {row['RF_Acc']:<6} {row['RF_Advantage']:<6} {row['DT_Overfit']:<7} {row['RF_Overfit']:<7} {row['Features']:<4} {row['Samples']:<6} {row['Classes']:<3}")

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

        if len(redundancy_scores) > 1:
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
            if row['Level'] in ['HIGH', 'MODERATE']:
                target_acc = float(row['RF_Acc']) + 0.02  # Should beat RF by 2%+
                print(f"     {row['Dataset']:<15}: Beat {row['RF_Acc']} (target: >{target_acc:.3f})")

        return summary_data

    def save_results_to_csv(self, filename="categorical_dataset_analysis.csv"):
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


# Run comprehensive analysis with baselines
if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE CATEGORICAL DATASET ANALYSIS")
    print("=" * 60)

    analyzer = ImprovedDatasetAnalyzer("C:/Users/David/PycharmProjects/Examensarbete/testsets")  # Update folder name  # Update folder name

    # Step 1: Analyze redundancy
    print("📊 STEP 1: ANALYZING REDUNDANCY...")
    results = analyzer.analyze_all_datasets()

    # Step 2: Run baseline experiments
    print("🏃 STEP 2: RUNNING BASELINE EXPERIMENTS...")
    baseline_results = analyzer.run_baseline_experiments()

    # Step 3: Generate comprehensive summary
    print("📋 STEP 3: GENERATING RESEARCH SUMMARY...")
    summary = analyzer.generate_baseline_summary()

    # Step 4: Save results
    analyzer.save_results_to_csv("categorical_dataset_analysis.csv")

    print(f"\n🎯 NEXT STEPS FOR YOUR DEEP FOREST:")
    print(f"1. Test your Deep Forest on these datasets")
    print(f"2. Focus on datasets with MODERATE or HIGH redundancy")
    print(f"3. Compare against the baseline accuracies shown above")
    print(f"4. Your Deep Forest should show strongest advantage on high-redundancy datasets")
    print(f"✅ ANALYSIS COMPLETE!")