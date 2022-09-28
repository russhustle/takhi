import matplotlib.pyplot as plt
import numpy as np

"""
1. Use the function below (plot_roc_curve)
2. Scikit-learn api: sklearn.metrics.RocCurveDisplay
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
3. Yellowbrick api
   https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html
4. Wandb api
   https://docs.wandb.ai/guides/integrations/scikit
"""


def plot_roc_curve(recalls, precisions, fpr, tpr, label=None):
    """ROC curve.
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
    plt.plot(fpr, tpr, "b-", linewidth=2, label=label)  # ROC curve
    recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
    plt.plot([fpr_90, fpr_90], [0.0, recall_90_precision], "r:")
    plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
    plt.plot([fpr_90], [recall_90_precision], "ro")  # Red line
    plt.text(
        fpr_90 + 0.02,
        recall_90_precision,
        f"A (precision=0.9, recall={recall_90_precision:.2f})",
    )  # Point A

    # General settings
    plt.plot([0, 1], [0, 1], "k--")  # diagonal line: random classifier
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=16)
    plt.ylabel("True Positive Rate (Recall)", fontsize=16)
    plt.grid(True)
    plt.title("ROC Curve. Point A highlights the chosen ratio")
    plt.show()
