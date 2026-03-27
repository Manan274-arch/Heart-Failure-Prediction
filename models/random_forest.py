import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing import X_train, X_test, y_train, y_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("===== Random Forest =====")

# Initialize model
model = RandomForestClassifier(
    n_estimators=500,          # number of trees
    max_depth=6,               # prevent over-complex trees
    min_samples_leaf=3,        # stabilize leaves
    class_weight="balanced",   # handle imbalance
    random_state=42,            
)


# Train
model.fit(X_train, y_train)

# Predict
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.30
y_pred = (y_prob >= threshold).astype(int)
# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", round(accuracy,2))
print("Precision:", round(precision,2))
print("Recall:", round(recall,2))
print("F1 Score:", round(f1,2))
print("ROC-AUC:", round(roc_auc,2))
