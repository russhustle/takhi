def confusion_matrix_heatmap(confusion_matrix, cmap="Reds"):
    """ Show confusion_matrix as heatmap.
    Args:
        confusion_matrix (_type_): _description_
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    hmap = sns.heatmap(data=confusion_matrix, annot=True, fmt="d", cmap=cmap)
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True")
    plt.xlabel("Predicted")
