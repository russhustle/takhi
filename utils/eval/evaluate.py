import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
- plot_precision_recall_vs_threshold
- plot_precision_vs_recall
- plot_roc_curve
- confusion_matrix_heatmap
"""


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, axis=50000):
    """Precision and recall versus the decision threshold
        source: https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
    Args:
        Get precisions, recalls, thresholds from sklearn.metrics.precision_recall_curve.
            e.g. precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
        axis (int, optional): Axis range. Defaults to 50000.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-axis, axis, -0.1, 1.1])
    recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    plt.plot([threshold_90_precision, threshold_90_precision], [0.0, 0.9], "r:")
    plt.plot([-axis, threshold_90_precision], [0.9, 0.9], "r:")
    plt.plot(
        [-axis, threshold_90_precision],
        [recall_90_precision, recall_90_precision],
        "r:",
    )
    plt.plot([threshold_90_precision], [0.9], "ro")
    plt.plot([threshold_90_precision], [recall_90_precision], "ro")
    plt.show()
