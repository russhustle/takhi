import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def kmeans_elbow_plot(data, num=(1, 10)):
    # https://predictivehacks.com/k-means-elbow-method-code-for-python/
    np.random.seed(42)
    distortions = []
    K = range(num[0], num[1])
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    plt.show()


def yb_kmeans_elbow(data, k=(1, 10)):
    # https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
    np.random.seed(42)
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(k[0], k[1]))
    visualizer.fit(data)
    visualizer.show()
