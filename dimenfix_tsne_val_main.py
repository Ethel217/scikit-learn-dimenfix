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
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# fix value on iris dataset

def main():

    np.random.seed(42)

    # # use wine dataset
    # wine = load_wine()
    # X = wine.data
    # label = wine.target.astype(int)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # use iris dataset
    iris = load_iris()
    X = iris.data
    label = iris.target.astype(int)
    X = preprocessing.MinMaxScaler().fit_transform(X)

    n_points = X.shape[0]
    print("Number of points:", n_points)

    # iris: fix Sepal width [1]
    sepal_width = X[:, 1]
    sepal_width = preprocessing.MinMaxScaler(feature_range=(0, 100)).fit_transform(sepal_width.reshape(-1, 1)).flatten()
    # hard fix to exact value
    range_limits = np.column_stack((sepal_width, sepal_width))
    # print("Range limits:\n", range_limits)

    # # wine: fix non-flavanoid_phenols [7]
    # non_flav = X[:, 1]
    # non_flav = preprocessing.MinMaxScaler(feature_range=(0, 100)).fit_transform(non_flav.reshape(-1, 1)).flatten()
    # # hard fix to exact value
    # range_limits = np.column_stack((non_flav, non_flav))

    print(range_limits.shape)

    start = timer()
    y = TSNEDimenfix(n_components=2, learning_rate='auto', init='random', perplexity=10, \
                     method="exact", \
                     range_limits=range_limits, dimenfix=True, class_ordering="disable", class_label=label, fix_iter=50, mode="clip", early_push=True).fit_transform(X)
    end = timer()
    # print(f"{trustworthiness(X, y, n_neighbors=20):.3f}")

    # print(np.amin(y, axis=0))

    print('Dimenfix TSNE with fixed value took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=label, cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.colorbar()
    plt.savefig('.\\figures\\iris_fix_val.png', dpi=300, bbox_inches='tight')
    plt.show()
    # plt.clf()
    return


if __name__ == "__main__":
    main()
    exit(0)
