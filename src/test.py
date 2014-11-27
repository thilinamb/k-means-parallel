from KMeansBase import KMeansBase
from KMeansPP import KMeansPP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == '__main__':
    k = 3
    #data = np.random.randn(100,2)
    data = np.array([[1.1,2],[1,2],[0.9,1.9],[1,2.1],[4,4],[4,4.1],[4.2,4.3],[4.3,4],[9,9],[8.9,9],[8.7,9.2],[9.1,9]])
    kmeans = KMeansPP(data, k)
    _, _, centroids, min_location = kmeans.cluster()
    # plotting code
    plt.figure()
    plt.subplot(1,2,1)
    colors = iter(cm.rainbow(np.linspace(0, 1, k + 1)))
    for col in range (0,k):
            plt.scatter(data[min_location[:,col] == True, :][:,0], data[min_location[:,col] == True, :][:,1], color=next(colors))

    centroid_leg = plt.scatter(centroids[:,0], centroids[:,1], color=next(colors), marker='x')
    plt.legend([centroid_leg], ['Centroids'], scatterpoints=1, loc='best')

    kmeans = KMeansBase(data, k)
    _, _, centroids, min_location = kmeans.cluster()
    plt.subplot(1,2,2)
    colors = iter(cm.rainbow(np.linspace(0, 1, k + 1)))
    for col in range (0,k):
            plt.scatter(data[min_location[:,col] == True, :][:,0], data[min_location[:,col] == True, :][:,1], color=next(colors))

    centroid_leg = plt.scatter(centroids[:,0], centroids[:,1], color=next(colors), marker='x')
    plt.legend([centroid_leg], ['Centroids'], scatterpoints=1, loc='best')

    plt.show()

