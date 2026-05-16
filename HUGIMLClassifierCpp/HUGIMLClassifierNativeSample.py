"""
HUGIMLClassifierNativeSample.py
================================
Sample script demonstrating HUGIMLClassifierNative on two benchmark datasets:
  - Pima Indians Diabetes  (all-numerical features)
  - Titanic                (mixed numerical + categorical features)

Each dataset is run under two configurations to illustrate the effect of
pattern length (L) and bin-count selection (B).

Run from the repository root:
    python HUGIMLClassifierNativeSample.py
"""

import time
import numpy as np
import pandas as pd

from HUGIMLClassifierNative import HUGIMLClassifierNative
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =============================================================================
# Helpers
# =============================================================================

def print_metrics(y_test, y_pred, y_prob, label="TEST METRICS"):
    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*48}")
    print(f"  {label}")
    print(f"{'='*48}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Balanced Accuracy  : {bacc:.4f}")
    print(f"  Precision          : {prec:.4f}")
    print(f"  Recall             : {rec:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"  ROC AUC            : {auc:.4f}")
    print(f"\n  Confusion Matrix")
    print(f"  {confusion_matrix(y_test, y_pred)}")
    print(f"\n  Classification Report")
    print(classification_report(y_test, y_pred, digits=4))


def print_patterns(clf, top_n=10):
    features = clf.get_hug_features()
    info     = clf.get_pattern_info()
    print(f"\n  Top {min(top_n, len(features))} mined HUG patterns "
          f"(of {len(features)} total):")
    print(f"  {'Pattern':<40} {'Utility':>8} {'IG':>8} {'Support':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in info.head(top_n).iterrows():
        print(f"  {row['pattern']:<40} {row['utility']:>8.4f} "
              f"{row['information_gain']:>8.4f} {row['support']:>8.4f}")


# =============================================================================
# Dataset 1 — Pima Indians Diabetes  (all-numerical features)
# =============================================================================

print("\n" + "#"*60)
print("# Dataset 1 : Pima Indians Diabetes")
print("#"*60)

cols = ['numPregnancies', 'glucose', 'bp', 'skinThickness',
        'insulin', 'bmi', 'diabetesPedigreeFunction', 'age', 'class']
pima = pd.read_csv('datasets/pima indians diabetes.csv', header=None, names=cols)

X_pima = pima.iloc[:, :-1]
y_pima = pima.iloc[:,  -1].values.astype(np.int64)

numericIntCols   = [c for c in X_pima.columns if np.issubdtype(X_pima[c].dtype, np.integer)]
numericFloatCols = [c for c in X_pima.columns if np.issubdtype(X_pima[c].dtype, float)]
catCols          = [c for c in X_pima.columns if X_pima[c].dtype == object]
allCols          = [numericIntCols, numericFloatCols, catCols]

x_tr, x_te, y_tr, y_te = train_test_split(
    X_pima, y_pima, test_size=0.2, random_state=0, stratify=y_pima
)

# ── Run 1 : singleton patterns, supervised bin selection ─────────────────────
print("\n--- Configuration: B=-1 (auto bins), L=1 (singletons) ---")

clf1 = HUGIMLClassifierNative(B=-1, L=1, G=1e-6,
                               allCols=allCols,
                               origColumns=X_pima.columns.tolist())
t0 = time.perf_counter()
clf1.fit(x_tr, y_tr)
fit_ms = (time.perf_counter() - t0) * 1000

y_pred_proba = clf1.predict_proba(x_te)
y_pred       = np.argmax(y_pred_proba, axis=1)
y_prob       = y_pred_proba[:, 1]

print(f"\n  Fit time : {fit_ms:.1f} ms   |   Patterns mined : {len(clf1.patterns_)}")
print_metrics(y_te, y_pred, y_prob, "Pima | B=-1, L=1")
print_patterns(clf1, top_n=8)

# ── Run 2 : pair patterns, fixed bins ────────────────────────────────────────
print("\n--- Configuration: B=5, L=2 (singleton + pair patterns) ---")

clf2 = HUGIMLClassifierNative(B=5, L=2, G=1e-4,
                               allCols=allCols,
                               origColumns=X_pima.columns.tolist())
t0 = time.perf_counter()
clf2.fit(x_tr, y_tr)
fit_ms = (time.perf_counter() - t0) * 1000

y_pred_proba2 = clf2.predict_proba(x_te)
y_pred2       = np.argmax(y_pred_proba2, axis=1)
y_prob2       = y_pred_proba2[:, 1]

print(f"\n  Fit time : {fit_ms:.1f} ms   |   Patterns mined : {len(clf2.patterns_)}")
print_metrics(y_te, y_pred2, y_prob2, "Pima | B=5, L=2")
print_patterns(clf2, top_n=8)

# ── Stratified K-Fold cross-validation ───────────────────────────────────────
print("\n--- 5-fold stratified cross-validation (B=-1, L=1) ---\n")

clf_cv = HUGIMLClassifierNative(B=-1, L=1, G=1e-6,
                                 allCols=allCols,
                                 origColumns=X_pima.columns.tolist())
cv = cross_validate(
    clf_cv, X_pima, y_pima,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['accuracy', 'balanced_accuracy', 'f1', 'roc_auc'],
    return_train_score=False,
)
print(f"  {'Metric':<22} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*40}")
for metric in ['test_accuracy', 'test_balanced_accuracy', 'test_f1', 'test_roc_auc']:
    vals = cv[metric]
    label = metric.replace('test_', '').replace('_', ' ').title()
    print(f"  {label:<22} {vals.mean():>8.4f} {vals.std():>8.4f}")


# =============================================================================
# Dataset 2 — Titanic  (mixed numerical + categorical features)
# =============================================================================

print("\n\n" + "#"*60)
print("# Dataset 2 : Titanic  (mixed feature types)")
print("#"*60)

titanic = pd.read_csv('datasets/titanic.csv')
titanic = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                   'Fare', 'Embarked', 'Survived']].dropna().reset_index(drop=True)

X_tit = titanic.drop(columns=['Survived'])
y_tit = titanic['Survived'].values.astype(np.int64)

# ── prepareXy path: auto-detects column types from DataFrame dtypes ──────────
# Call prepareXy on the full dataset first (type detection only), then split.
# This is the recommended Path A pattern: type masks are set on the classifier
# and reused consistently for both fit() and predict_proba().

print("\n--- Configuration: B=7, L=1, using prepareXy for type detection ---")

clf3 = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
X_prep, y_prep = clf3.prepareXy(X_tit, y_tit)

X_tr_prep, X_te_prep, y_tr_t, y_te_t = train_test_split(
    X_prep, y_prep, test_size=0.2, random_state=0, stratify=y_prep
)

t0 = time.perf_counter()
clf3.fit(X_tr_prep, y_tr_t)
fit_ms = (time.perf_counter() - t0) * 1000

y_pred_proba3 = clf3.predict_proba(X_te_prep)
y_pred3       = np.argmax(y_pred_proba3, axis=1)
y_prob3       = y_pred_proba3[:, 1]

print(f"\n  Fit time : {fit_ms:.1f} ms   |   Patterns mined : {len(clf3.patterns_)}")
print_metrics(y_te_t, y_pred3, y_prob3, "Titanic | B=7, L=1")
print_patterns(clf3, top_n=10)

# ── Pattern info table ───────────────────────────────────────────────────────
print("\n  Full pattern info table:")
print(clf3.get_pattern_info().to_string(index=False))

# ── L=2 compound patterns ─────────────────────────────────────────────────────
print("\n--- Configuration: B=5, L=2, compound patterns ---")

clf4 = HUGIMLClassifierNative(B=5, L=2, G=1e-4)
X_prep4, y_prep4 = clf4.prepareXy(X_tit, y_tit)

X_tr_prep4, X_te_prep4, y_tr_t4, y_te_t4 = train_test_split(
    X_prep4, y_prep4, test_size=0.2, random_state=0, stratify=y_prep4
)

t0 = time.perf_counter()
clf4.fit(X_tr_prep4, y_tr_t4)
fit_ms = (time.perf_counter() - t0) * 1000

y_pred_proba4 = clf4.predict_proba(X_te_prep4)
y_pred4       = np.argmax(y_pred_proba4, axis=1)
y_prob4       = y_pred_proba4[:, 1]

print(f"\n  Fit time : {fit_ms:.1f} ms   |   Patterns mined : {len(clf4.patterns_)}")
print_metrics(y_te_t4, y_pred4, y_prob4, "Titanic | B=5, L=2")
print_patterns(clf4, top_n=10)
