import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from sklearn.manifold import TSNEDimenfix
from sklearn.datasets import fetch_openml
from sklearn.manifold import trustworthiness

def main():
    # input dataset
    mnist = fetch_openml('mnist_784', version=1, data_home=".\\scikit_learn_data")
    X = mnist.data.to_numpy()
    sample_indices = np.random.choice(X.shape[0], size=1000, replace=False)
    X = X[sample_indices]

    label = mnist.target.to_numpy()
    label = label[sample_indices].astype(int)

    X = preprocessing.MinMaxScaler().fit_transform(X)
    n_points = X.shape[0]
    print("Number of points:", n_points)

    # create movable range limits for each point: as percentage!!
    unique_labels, counts = np.unique(label, return_counts=True)
    n_labels = len(unique_labels)
    print(f'Number of different labels: {n_labels}')
    print(f'Counts per label: {counts}')

    # equally band ver
    range_width = 100 / n_labels
    range_limits = np.zeros((X.shape[0], 2))
    for l in unique_labels:
        range_limits[label == l] = [l * range_width, (l + 1) * range_width]
    print(range_limits.shape)

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
                    #  method="exact", \
                      dimenfix=True, range_limits=range_limits, class_ordering=False, class_label=label, fix_iter=50, mode="gaussian", early_push=False).fit_transform(X)
    end = timer()
    # print(f"{trustworthiness(X, y, n_neighbors=20):.3f}")

    print(np.amin(y, axis=0))

    print('Dimenfix TSNE with class ordering took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=label, cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.colorbar()
    plt.grid(linestyle='dotted')
    plt.savefig('.\\figures\\mnist_1000_band_fix.png', dpi=300, bbox_inches='tight')
    plt.show()
    # plt.clf()
    
    return

if __name__ == "__main__":
    main()
    exit(0)
