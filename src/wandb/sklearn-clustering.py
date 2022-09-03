import wandb
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans

"""
- elbow curve
- Log silhouette plot

Four steps:
1. Login
2. Dataset
3. Model
4. Wandb
"""
# 1. Login
wandb.login()

# 2. Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target
names = iris.target_names

def get_label_ids(classes):
    return np.array([names[aclass] for aclass in classes])
labels = get_label_ids(y)

# 3. Train model
kmeans = KMeans(n_clusters=4, random_state=1)
cluster_labels = kmeans.fit_predict(X)

# 4. Wandb
run = wandb.init(project='sklearn-trial', name="clustering")
wandb.sklearn.plot_clusterer(kmeans, X, cluster_labels, labels, 'KMeans')
wandb.finish()
