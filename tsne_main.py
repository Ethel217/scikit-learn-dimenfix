import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
# from dimenfix_force_scheme import DimenFixForceScheme
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# test if scikitlearn is installed correctly


def main():

    np.random.seed(42)

    # Wine dataset
    wine = load_wine()
    X = wine.data
    label = wine.target.astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # MNIST dataset
    # mnist = fetch_openml('mnist_784', version=1, data_home=".\\scikit_learn_data")
    # X = mnist.data.to_numpy()
    # sample_indices = np.random.choice(X.shape[0], size=1000, replace=False)
    # X = X[sample_indices]
    # label = mnist.target.to_numpy()
    # label = label[sample_indices].astype(int)
    # X = preprocessing.MinMaxScaler().fit_transform(X)

    n_points = X.shape[0]
    print("Number of points:", n_points)

    start = timer()
    y = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10, method="exact").fit_transform(X)
    # y = DimenFixForceScheme(max_it=1000, fixed_feature=fixed_feature, alpha=0.8).fit_transform(X)
    end = timer()
    # print(f"{trustworthiness(X, y, n_neighbors=20):.3f}")

    print(np.amin(y, axis=0))

    print('Regular TSNE took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=label, cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.colorbar()
    plt.savefig('.\\figures\\wine_regular_tsne_color_class.png', dpi=300, bbox_inches='tight')
    plt.show()

    return


if __name__ == "__main__":
    main()
    exit(0)
