import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
# from dimenfix_force_scheme import DimenFixForceScheme
from sklearn.manifold import TSNEDimenfix
from sklearn.datasets import fetch_openml
from sklearn.manifold import trustworthiness
from sklearn.datasets import load_iris

# fix value on iris dataset

def main():

    # input iris dataset
    iris = load_iris()
    X = iris.data
    label = iris.target.astype(int)

    X = preprocessing.MinMaxScaler().fit_transform(X)
    n_points = X.shape[0]
    print("Number of points:", n_points)

    # fix Sepal width [1]
    sepal_width = X[:, 1]
    sepal_width = preprocessing.MinMaxScaler(feature_range=(0, 100)).fit_transform(sepal_width.reshape(-1, 1)).flatten()
    # hard fix to exact value
    range_limits = np.column_stack((sepal_width, sepal_width))
    # print("Range limits:\n", range_limits)
    print(range_limits.shape)

    start = timer()
    y = TSNEDimenfix(n_components=2, learning_rate='auto', init='random', perplexity=10, early_exaggeration=4, max_iter=500, range_limits=range_limits, dimenfix=True, class_ordering=False, class_label=label, fix_iter=30, mode="clip").fit_transform(X)
    end = timer()
    # print(f"{trustworthiness(X, y, n_neighbors=20):.3f}")

    # print(np.amin(y, axis=0))

    print('Dimenfix TSNE with fixed value took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=label, cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.colorbar()
    plt.grid(linestyle='dotted')
    plt.savefig('.\\figures\\iris_fix_val.png', dpi=300, bbox_inches='tight')
    plt.show()
    # plt.clf()
    
    # start = timer()
    # y = TSNEDimenfix(n_components=2, learning_rate='auto', init='random', perplexity=10, range_limits=range_limits).fit_transform(X)
    # # y = DimenFixForceScheme(max_it=1000, fixed_feature=fixed_feature, alpha=0.8).fit_transform(X)
    # end = timer()
    # print(f"{trustworthiness(X, y, n_neighbors=20):.3f}")

    # print(np.amin(y, axis=0))

    # print('Dimenfix TSNE took {0} to execute'.format(timedelta(seconds=end - start)))

    # plt.figure()
    # plt.scatter(y[:, 1], y[:, 0], c=label, cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    # plt.colorbar()
    # plt.grid(linestyle='dotted')
    # plt.savefig('.\\figures\\mnist_600_fix.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.clf()
    
    # start = timer()
    # y = TSNEDimenfix(n_components=2, learning_rate='auto', init='random', perplexity=10, range_limits=None).fit_transform(X)
    # end = timer()

    # print(f"{trustworthiness(X, y, n_neighbors=20):.3f}")

    # print(np.amin(y, axis=0))

    # print('Regular TSNE took {0} to execute'.format(timedelta(seconds=end - start)))

    # plt.figure()
    # plt.scatter(y[:, 1], y[:, 0], c=label, cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    # plt.grid(linestyle='dotted')
    # plt.colorbar()
    # plt.savefig('.\\figures\\mnist_600_base.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return


if __name__ == "__main__":
    main()
    exit(0)
