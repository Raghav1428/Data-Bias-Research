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
                    lambda g: recall_score(y_true[g.index], y_pred[g.index], average='macro', zero_division=0))
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

    def visualize_fairness(self):
        spd_records = []
        eod_records = []
        tpr_matrix = {}

        for model, metrics in self.fairness_metrics.items():
            for attr, values in metrics.items():
                spd_records.append((model, values['statistical_parity_difference']))
                eod_records.append((model, values['equal_opportunity_difference']))
                tpr_matrix[model] = values['group_metrics']['TPR']

        # Bar plots for SPD and EOD
        spd_df = pd.DataFrame(spd_records, columns=['Model', 'SPD'])
        eod_df = pd.DataFrame(eod_records, columns=['Model', 'EOD'])

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=spd_df, x='Model', y='SPD', ax=axs[0], palette="rocket")
        axs[0].set_title("Bias Across Models: Higher SPD Means Unequal Selection")
        axs[0].set_ylabel("Statistical Parity Difference (SPD)")
        axs[0].set_xlabel("Model")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")
        axs[0].text(-0.5, max(spd_df['SPD']) * 0.95, "A fair model should have SPD close to 0", fontsize=9)

        sns.barplot(data=eod_df, x='Model', y='EOD', ax=axs[1], palette="mako")
        axs[1].set_title("Opportunity Bias: Higher EOD Means Recall Varies by Group")
        axs[1].set_ylabel("Equal Opportunity Difference (EOD)")
        axs[1].set_xlabel("Model")
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")
        axs[1].text(-0.5, max(eod_df['EOD']) * 0.95, "Lower is fairer (uniform recall)", fontsize=9)

        plt.suptitle("Fairness Metrics Across ML Models\n(Helps reveal if a model treats different journals equally)", fontsize=12, y=1.05)
        plt.tight_layout()
        plt.savefig("fairness_barplots.png")
        plt.show()

        # Heatmap of TPR per journal
        tpr_df = pd.DataFrame(tpr_matrix).T
        plt.figure(figsize=(10, 6))
        sns.heatmap(tpr_df, annot=True, cmap="viridis", fmt=".2f")
        plt.title("TPR Heatmap: Model Accuracy Per Journal Group\n(Darker cells mean better recall for that journal)", fontsize=12)
        plt.ylabel("Model")
        plt.xlabel("Journal")
        plt.tight_layout()
        plt.savefig("fairness_tpr_heatmap.png")
        plt.show()


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
    ml.visualize_fairness()
