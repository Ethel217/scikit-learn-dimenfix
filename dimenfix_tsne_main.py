import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from sklearn.manifold import TSNEDimenfix
from sklearn.datasets import fetch_openml
from sklearn.manifold import trustworthiness

import json
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

def main():

    np.random.seed(42)

    # use wine dataset
    # wine = load_wine()
    # X = wine.data
    # label = wine.target.astype(int)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # use breast cancer dataset
    # bc = load_breast_cancer()
    # X = bc.data
    # X = preprocessing.MinMaxScaler().fit_transform(X)
    # label = bc.target.astype(int)

    # use MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, data_home=".\\scikit_learn_data")
    X = mnist.data.to_numpy()
    sample_indices = np.random.choice(X.shape[0], size=5000, replace=False)
    X = X[sample_indices]
    X = preprocessing.MinMaxScaler().fit_transform(X)
    label = mnist.target.to_numpy()
    label = label[sample_indices].astype(int)

    n_points = X.shape[0]
    print("Number of points:", n_points)

    # create movable range limits for each point: as percentage!!
    unique_labels, counts = np.unique(label, return_counts=True)
    n_labels = len(unique_labels)
    print(f'Number of different labels: {n_labels}')
    print(f'Counts per label: {counts}')

    # equal bands ver
    range_width = 100 / n_labels
    range_limits = np.zeros((X.shape[0], 2))
    for l in unique_labels:
        range_limits[label == l] = [l * range_width, (l + 1) * range_width]
    # print(range_limits.shape)

    # band scale with class density ver
    # total_count = counts.sum()
    # proportions = counts / total_count

    # cumulative_ranges = np.cumsum(proportions) * 100
    # range_limits = np.zeros((X.shape[0], 2))

    # start = 0
    # for i, l in enumerate(unique_labels):
    #     end = cumulative_ranges[i]
    #     range_limits[label == l] = [start, end]
    #     start = end

    start = timer()
    y = TSNEDimenfix(n_components=2, learning_rate='auto', init='random', perplexity=10, \
                     method="exact", \
                      dimenfix=True, range_limits=range_limits, density_adj=False, class_ordering="p_sim", class_label=label, fix_iter=50, mode="rescale", early_push=False).fit_transform(X)
    end = timer()
    # print(f"{trustworthiness(X, y, n_neighbors=20):.3f}")

    print(np.amin(y, axis=0))

    print('Dimenfix TSNE with class ordering took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=label, cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    pic_size = np.max(y[:, 0]) - np.min(y[:, 0])
    plt.ylim(np.min(y[:, 0]) - pic_size * 0.01, np.max(y[:, 0]) + pic_size * 0.01)
    plt.colorbar()
    plt.savefig('.\\figures\\MNIST_fix_class.png', dpi=300, bbox_inches='tight')
    plt.show()
    # plt.clf()

    # y = y.tolist()
    embedding = [{"x": float(y_), "y": float(x_)} for x_, y_ in y]
    with open('.\\visualization\\embedding.json', 'w') as f:
        json.dump(embedding, f, indent=2)
    
    return

if __name__ == "__main__":
    main()
    exit(0)
