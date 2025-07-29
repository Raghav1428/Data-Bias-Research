import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

def load_cord19_dataset():
    try:
        df = pd.read_csv("data/cord19/all_sources_metadata_2020-03-13.csv", low_memory=False)
        df = df[['title', 'abstract', 'publish_time', 'journal']].dropna()
        df = df[df['abstract'].str.len() > 50]
        print("✅ CORD-19 dataset loaded:", df.shape)
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

class BiasAwareMLFramework:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        self.results = {}
        self.fairness_metrics = {}

    def train_models(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train.toarray(), y_train)
            y_pred = model.predict(X_test.toarray())

            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test.toarray())
                if len(set(y_test)) > 2:
                    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
                    auc = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovr')
                else:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                y_pred_proba = y_pred
                auc = float('nan')

            self.results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'auc': auc,
                'predictions': y_pred
            }

            print(f"{name} - Accuracy: {self.results[name]['accuracy']:.3f}, AUC: {self.results[name]['auc']:.3f}")

    def calculate_fairness_metrics(self, y_test, sensitive_features):
        for name, res in self.results.items():
            y_pred = pd.Series(res['predictions']).reset_index(drop=True)
            y_true = pd.Series(y_test).reset_index(drop=True)
            fairness_scores = {}
            for feature_name in sensitive_features.columns:
                groups = sensitive_features[feature_name].reset_index(drop=True)
                sp = y_pred.groupby(groups).mean()
                tpr = groups.groupby(groups).apply(
                    lambda g: recall_score(y_true[g.index], y_pred[g.index], average='macro', zero_division=0)
)
                parity = tpr.max() - tpr.min() if len(tpr) > 1 else 0
                fairness_scores[feature_name] = {
                    'statistical_parity_difference': sp.max() - sp.min() if len(sp) > 1 else 0,
                    'equal_opportunity_difference': parity,
                    'group_metrics': {'TPR': tpr.to_dict(), 'SP': sp.to_dict()}
                }
            self.fairness_metrics[name] = fairness_scores

    def generate_report(self):
        print("\n=== MODEL PERFORMANCE SUMMARY ===")
        df = pd.DataFrame({
            name: {
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1': res['f1'],
                'AUC': res['auc']
            } for name, res in self.results.items()
        }).T
        print(df.round(3))

        print("\n=== FAIRNESS METRICS ===")
        for model, fair in self.fairness_metrics.items():
            print(f"\n{model}:")
            for attr, metrics in fair.items():
                print(f"  {attr} SPD: {metrics['statistical_parity_difference']:.4f}, EOD: {metrics['equal_opportunity_difference']:.4f}")

if __name__ == "__main__":
    print("\n🏥 Medical AI Bias Analysis Framework\n" + "=" * 50)

    df = load_cord19_dataset()
    if df is None:
        exit()

    top_journals = df['journal'].value_counts().nlargest(5).index
    df = df[df['journal'].isin(top_journals)].copy()

    X = TfidfVectorizer(max_features=1000).fit_transform(df['abstract'])
    le_journal = LabelEncoder()
    y = le_journal.fit_transform(df['journal'])

    sensitive_attr = df['journal']

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_attr, test_size=0.2, stratify=y, random_state=42
    )

    ml = BiasAwareMLFramework()
    ml.train_models(X_train, X_test, y_train, y_test)
    ml.calculate_fairness_metrics(y_test, pd.DataFrame({'journal': sens_test}))
    ml.generate_report()