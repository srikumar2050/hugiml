import numpy as np
import pandas as pd

from HUGIMLClassifierPy import HUGIMLClassifierPy
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

fname='datasets/pima-indians-diabetes.csv'
data = pd.read_csv(fname, header=None)
data.columns = ['numPregnancies', 'glucose', 'bp', 'skinThickness', 'insulin', 'bmi', 'diabetesPedigreeFunction', 'age', 'class']
X, y = data.iloc[:,:-1], data.iloc[:,-1]
numericColumns, catColumns = X.columns.tolist(), []

numericIntCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, np.integer)]
numericFloatCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, float)]
catCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]
allCols = [numericIntCols, numericFloatCols, catColumns]

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

params = {
    'B': -1,
    'L': 1,
    'G': 1e-6,
    'allCols': allCols, 'origColumns': X.columns.tolist()
}

clf = HUGIMLClassifierPy(**params)

clf.fit(x_train, y_train)

y_pred_proba = clf.predict_proba(x_test)
y_prob = y_pred_proba[:, 1]
y_pred = np.argmax(y_pred_proba, axis=1)

acc  = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)

print("\n================ TEST METRICS ================")

print(f"Accuracy           : {acc:.4f}")
print(f"Balanced Accuracy  : {bacc:.4f}")
print(f"Precision          : {prec:.4f}")
print(f"Recall             : {rec:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC AUC            : {auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))

print(clf.get_hug_features())               # human-readable pattern names
print(clf.get_pattern_info())               # utility / IG / support table
