Common Functions in Machine Learning Projects
===

[TOC]

- DT_Vis
    - Decision tree visualization after fitting
- kmeans
    - kmeans_elbow_plot
    - yb_kmeans_elbow

Classification
---

[wandb.sklearn.plot_classifier](https://docs.wandb.ai/guides/integrations/scikit) (recommend)

[yellowbrick.classifier](https://www.scikit-yb.org/en/latest/api/classifier/index.html)

### 01.ROC Plot

1. [code](https://github.com/Sihan-A/takhi/blob/main/src/evaluate.py)
2. [sklearn.metrics.RocCurveDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay)
3. [yellowbrick.classifier.ROCAUC](https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html)

### 02.Confusion Matrix

Decision Tree
---

Decision tree visualization: [`sklearn.tree.plot_tree`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)

Clustering
---

[yellowbrick.cluster](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html)

- `KElbowVisualizer`: K-Means Elbow method visualization
