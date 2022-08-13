import wandb
from sklearn import datasets
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

"""
1. Log in to weights & biases
2. Dataset preparation
3. Model fitting
4. Save
"""
# 1. Login
wandb.login()

# 2. Dataset
housing = datasets.fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. Train model, get predictions
reg = Ridge()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# 4. Wandb
run = wandb.init(project='sklearn-trial', name="regression")
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name='Ridge')
wandb.finish()
