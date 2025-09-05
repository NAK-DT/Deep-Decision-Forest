import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, fetch_openml
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

warnings.filterwarnings('ignore')


class MultiDatasetAnalyzer:
    def __init__(self):
        self.results = {}

    def load_diabetes(self):
        ds = fetch_openml(name='diabetes', version=1, as_frame=True)
        X = ds.data.to_numpy(dtype=np.float64)
        y = LabelEncoder().fit_transform(ds.target)
        return X, y, 'diabetes'

    def load_mushroom(self):
        ms = fetch_openml('mushroom', version=1, as_frame=True)
        X = pd.get_dummies(ms.data.replace('?', np.nan)).fillna(0)
        y = (ms.target == 'p').astype(int)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(X) - 2000, random_state=42)
        keep_idx, _ = next(sss.split(X, y))
        X_sub, y_sub = X.iloc[keep_idx], y.iloc[keep_idx]
        data = np.hstack([X_sub.values, y_sub.values.reshape(-1, 1)])
        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1].astype(int)
        return X, y, 'mushroom'

    def load_heart(self):
        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        y = heart.data.targets.copy()
        X = X.apply(pd.to_numeric, errors='coerce')
        y = y.apply(pd.to_numeric, errors='coerce')
        df = pd.concat([X, y], axis=1).dropna().reset_index(drop=True)
        target_col = y.columns[0]
        if df[target_col].nunique() > 2 or df[target_col].max() > 1:
            df[target_col] = (df[target_col] > 0).astype(int)
        else:
            df[target_col] = df[target_col].astype(int)
        X_np = df.drop(columns=[target_col]).to_numpy(dtype=np.float64)
        X_min = X_np.min(axis=0)
        X_max = X_np.max(axis=0)
        den = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
        X_np = (X_np - X_min) / den
        y_np = df[target_col].to_numpy(dtype=np.int64)
        data = np.hstack([X_np, y_np.reshape(-1, 1)])
        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1]
        return X, y, 'heart_disease'

    def load_iris(self):
        data = load_iris()
        X, y = data.data, data.target
        return X, y, 'iris'

    def load_banknote(self):
        bank = fetch_openml(name='banknote-authentication', version=1, as_frame=True)
        X = bank.data.to_numpy()
        y = LabelEncoder().fit_transform(bank.target)
        return X, y, 'banknote_authentication'

    def load_maternal_health(self):
        maternal = fetch_ucirepo(id=863)
        X = maternal.data.features
        y = maternal.data.targets
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.values.ravel())
        data = np.hstack([X.values, y_encoded.reshape(-1, 1)])
        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1]
        return X.astype(np.float64), y.astype(int), 'maternal_health_risk'

    def load_raisin(self):
        ds = fetch_ucirepo(id=850)
        X = ds.data.features
        y = LabelEncoder().fit_transform(ds.data.targets.values.ravel())
        return X.to_numpy(), y, 'raisin'

    def calculate_redundancy_score(self, X):
        X = X.astype(np.float64)
        n_features = X.shape[1]
        stds = X.std(axis=0)
        if np.any(stds == 0):
            X = X[:, stds > 0]
            n_features = X.shape[1]
        corr_matrix = np.corrcoef(X.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        high_corr_count, total_pairs = 0, 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > 0.7:
                    high_corr_count += 1
                total_pairs += 1
        corr_score = high_corr_count / max(total_pairs, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        components_needed = np.argmax(cumvar >= 0.95) + 1
        pca_score = max(0, 1 - (components_needed / n_features))
        final_score = 0.5 * corr_score + 0.5 * pca_score
        return final_score, corr_score, pca_score

    def analyze(self):
        loaders = [
            self.load_diabetes,
            self.load_mushroom,
            self.load_heart,
            self.load_iris,
            self.load_banknote,
            self.load_maternal_health,
            self.load_raisin
        ]

        for loader in loaders:
            try:
                X, y, name = loader()
                r_total, r_corr, r_pca = self.calculate_redundancy_score(X)
                print(f"{name} -> corr: {r_corr:.3f}, pca: {r_pca:.3f}")
                level = 'HIGH' if r_total >= 0.6 else 'MODERATE' if r_total >= 0.3 else 'LOW'
                self.results[name] = {
                    'samples': len(X),
                    'features': X.shape[1],
                    'classes': len(np.unique(y)),
                    'redundancy_score': r_total,
                    'correlation_score': r_corr,
                    'pca_score': r_pca,
                    'level': level
                }
                print(f"✅ {name}: {r_total:.3f} ({level})")
            except Exception as e:
                print(f"❌ Failed on {loader.__name__}: {e}")

    def print_summary(self):
        print("\n📋 REDUNDANCY SUMMARY")
        print(f"{'Dataset':<25}{'Samples':<10}{'Features':<10}{'Redundancy':<15}{'Level':<10}")
        print("-" * 70)
        for name, info in sorted(self.results.items(), key=lambda x: x[1]['redundancy_score'], reverse=True):
            print(f"{name:<25}{info['samples']:<10}{info['features']:<10}{info['redundancy_score']:<15.3f}{info['level']:<10}")


if __name__ == '__main__':
    analyzer = MultiDatasetAnalyzer()
    analyzer.analyze()
    analyzer.print_summary()
