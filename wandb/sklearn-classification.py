import wandb
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

"""
1. Log feature importance
2. Log learning curve
3. Log confusion matrix
4. Log summary metrics
5. Log class proportions
5. Log calibration curve
6. Log roc curve
7. Log precision recall curve
"""

# 1. Login
wandb.login()

# 2. Load data
wbcd = wisconsin_breast_cancer_data = datasets.load_breast_cancer()
feature_names = wbcd.feature_names
labels = wbcd.target_names
X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2)

# 3. Train model, get predictions
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# 4. Wandb
run = wandb.init(project='sklearn-trial', name="classification")
wandb.sklearn.plot_classifier(
    model, 
    X_train, X_test, 
    y_train, y_test, 
    y_pred, y_probas, 
    labels, 
    is_binary=True, 
    model_name='RandomForest',
)
wandb.finish()
