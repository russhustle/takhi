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
    """ Precision and recall versus the decision threshold
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
    plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
    plt.plot([-axis, threshold_90_precision], [0.9, 0.9], "r:")
    plt.plot([-axis, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
    plt.plot([threshold_90_precision], [0.9], "ro")
    plt.plot([threshold_90_precision], [recall_90_precision], "ro")
    plt.show()

def plot_precision_vs_recall(precisions, recalls):
    """ Precision vs Recall.
        source: https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
    Args:
        Get precisions, recalls, thresholds from sklearn.metrics.precision_recall_curve.
            e.g. precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.title("Precision vs Recall. Point A shows when precision=0.9")
    plt.grid(True)
    recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
    plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
    plt.plot([recall_90_precision], [0.9], "ro")
    plt.text(recall_90_precision, 0.93, "A", fontsize=16)
    plt.show()

def plot_roc_curve(recalls, precisions, fpr, tpr, label=None):
    """ ROC curve.
        sklearn.metrics.plot_roc_curve
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html
    Args:
        Get precisions, recalls, thresholds from sklearn.metrics.precision_recall_curve.
            e.g. precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
        Get fpr, tpr from sklearn.metrics.roc_curve.
            e.g. fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    """
    plt.figure(figsize=(8, 6))

    # If we want to compare two models' roc curve. Add few lines like this.
    plt.plot(fpr, tpr, "b-", linewidth=2, label=label) # ROC curve
    recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
    plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
    plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
    plt.plot([fpr_90], [recall_90_precision], "ro") # Red line
    plt.text(fpr_90+0.02, recall_90_precision, f"A (precision=0.9, recall={recall_90_precision:.2f})") # Point A
    
    # General settings
    plt.plot([0, 1], [0, 1], 'k--') # diagonal line: random classifier
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)
    plt.title(f"ROC Curve. Point A highlights the chosen ratio")
    plt.show()

def confusion_matrix_heatmap(confusion_matrix, cmap="Reds"):
    """ Show confusion_matrix as heatmap.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    Args:
        confusion_matrix (_type_): _description_
    """
    hmap = sns.heatmap(data=confusion_matrix, annot=True, fmt="d", cmap=cmap)
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()
