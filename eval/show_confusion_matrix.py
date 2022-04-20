import seaborn as sns
import matplotlib.pyplot as plt

def show_confusion_matrix(confusion_matrix):
    """ Show confusion_matrix as heatmap.
    Args:
        confusion_matrix (_type_): _description_
    """
    hmap=sns.heatmap(data=confusion_matrix, annot=True, fmt="d", cmap="Reds")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True")
    plt.xlabel("Predicted")
