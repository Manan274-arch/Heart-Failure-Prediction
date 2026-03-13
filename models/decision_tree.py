import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing import X_train, X_test, y_train, y_test

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("===== Decision Tree =====")

# Model with basic regularization
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=3,
    min_samples_split=10,
    criterion="gini",
    class_weight="balanced",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)


