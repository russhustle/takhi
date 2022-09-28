import matplotlib.pyplot as plt
import numpy as np

"""
To plot precison vs recall
1. Use the code below.
2. Sklearn: sklearn.metrics.PrecisionRecallDisplay
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html
3. Yellowbrick: yellowbrick.classifier.PrecisionRecallCurve
   https://www.scikit-yb.org/en/latest/api/classifier/prcurve.html
4. W&B: wandb.sklearn.plot_precision_recall ⭐️
   https://docs.wandb.ai/guides/integrations/scikit
"""


def plot_precision_vs_recall(precisions, recalls):
    """Precision vs Recall.
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
    plt.plot([recall_90_precision, recall_90_precision], [0.0, 0.9], "r:")
    plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
    plt.plot([recall_90_precision], [0.9], "ro")
    plt.text(recall_90_precision, 0.93, "A", fontsize=16)
    plt.show()
