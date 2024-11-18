# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# This is the exact and Barnes-Hut t-SNE implementation. There are other
# modifications of the algorithm:
# * Fast Optimization for t-SNE:
#   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf

import warnings
from numbers import Integral, Real
from time import time
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform

from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..decomposition import PCA
from ..metrics.pairwise import _VALID_METRICS, pairwise_distances
from ..neighbors import NearestNeighbors
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.validation import _num_samples, check_non_negative, validate_data

# mypy error: Module 'sklearn.manifold' has no attribute '_utils'
# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from . import _barnes_hut_tsne, _utils  # type: ignore

MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P


def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose
    )
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))
    return P


def _kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    skip_num_points=0,
    compute_error=True,
):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _kl_divergence_bh(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    angle=0.5,
    skip_num_points=0,
    verbose=False,
    compute_error=True,
    num_threads=1,
):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2).

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : sparse matrix of shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized. Matrix should be of CSR format.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    angle : float, default=0.5
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    verbose : int, default=False
        Verbosity level.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    num_threads : int, default=1
        Number of threads used to compute the gradient. This is set here to
        avoid calling _openmp_effective_n_threads for each gradient step.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(
        val_P,
        X_embedded,
        neighbors,
        indptr,
        grad,
        angle,
        n_components,
        verbose,
        dof=degrees_of_freedom,
        compute_error=compute_error,
        num_threads=num_threads,
    )
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad

def accumulate_move_force(p, range_limits, class_label, x_range, out=False):
    p = p.reshape(range_limits[:, 0].shape[0], 2)

    # use clipping to estimate force: + up - down
    clipping_amounts = np.zeros(p.shape[0])
    unique_labels = np.unique(class_label)

    force_class_norm = np.zeros((len(unique_labels), 2)) # normalized force per class
    class_range_ori = np.zeros((len(unique_labels), 3))

    # accumulated "force" for each label
    for label in unique_labels:

        label_indices = np.where(class_label == label)[0]
        
        min_bound = np.min(p[:, 1]) + (range_limits[label_indices[0], 0] / 100) * x_range
        max_bound = np.min(p[:, 1]) + (range_limits[label_indices[0], 1] / 100) * x_range
        
        class_range_ori[label, 0] = label
        class_range_ori[label, 1] = range_limits[label_indices[0], 0]
        class_range_ori[label, 2] = range_limits[label_indices[0], 1]

        original_values = p[label_indices, 0]
        clipped_values = np.clip(original_values, min_bound, max_bound)
        clipping_amounts[label_indices] = original_values - clipped_values

        # class force average
        # TODO: not sure about normalization here by x_range
        force_class_norm[label] = [label, np.mean(clipping_amounts[label_indices])]
        
        # ?DONT Update p
        # p[label_indices, 0] = clipped_values
    class_range_ori = class_range_ori[np.argsort(class_range_ori[:, 1])]
    force_class_norm = force_class_norm[np.argsort(force_class_norm[:, 1])]

    class_scores = np.zeros((len(np.unique(class_label)), 2))
    for i, idx in enumerate(class_range_ori):
        class_scores[i, 0] = class_range_ori[i, 0]
        class_scores[i, 1] = i
    for i, idx in enumerate(force_class_norm):
        # TODO: adjust this divide by
        class_scores[int(force_class_norm[i][0]), 1] += (i - len(unique_labels) // 2) / 3
    class_scores = class_scores[np.argsort(class_scores[:, 1])]
    # print(class_scores)
    # print(class_range_ori)
    if out:
        print(force_class_norm)
    class_range_new = class_range_ori.copy()
    class_range_new[:, 0] = class_scores[:, 0]
    # print(class_range_new)
    range_limits_new = np.zeros(range_limits.shape)
    for i in range(range_limits.shape[0]):
        label = class_label[i]
        min_bound, max_bound = class_range_new[class_range_new[:, 0] == label, 1:3].flatten()
        range_limits_new[i, :] = [min_bound, max_bound]
    range_limits = range_limits_new
    # print(class_label[0])
    # print(range_limits[0, :])
    # print(range_limits_new[0, :])
    return range_limits

def _gradient_descent(
    objective,
    p0,
    it,
    max_iter,
    range_limits,
    class_ordering,
    class_label,
    fix_iter, # apply fix (and order classes) every N iters
    mode="clip",
    n_iter_check=1,
    n_iter_without_progress=300,
    momentum=0.8,
    learning_rate=200.0,
    min_gain=0.01,
    min_grad_norm=1e-7,
    verbose=0,
    args=None,
    kwargs=None,
):
    # TODO: embedding changes in this step
    # print("gradient decent run")
    
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    tic = time()
    for i in range(it, max_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == max_iter - 1

        error, grad = objective(p, *args, **kwargs)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        # TODO: dimenfix
        # clip (fix dimension) and re-order every fix_iter iters
        # if i != 0 and i % fix_iter == 0 and range_limits is not None:
            
        #     # print(range_limits[:, 0].shape[0])
        #     p = p.reshape(range_limits[:, 0].shape[0], 2)

        #     # plot and save
        #     p_ = p.copy()
        #     plotIntermediate(p_.ravel(), it=i, label=class_label, save=True, name="init")

        #     # resize as x-axis version
        #     current_range = np.max(p[:, 0]) - np.min(p[:, 0])
        #     x_range = (np.max(p[:, 1]) - np.min(p[:, 1]))
        #     scaling_factor = x_range / current_range if current_range != 0 else 1

        #     # rescale, also align with center at 0
        #     p[:, 0] = (p[:, 0] - np.mean(p[:, 0])) * scaling_factor

        #     # plot rescaled
        #     p_ = p.copy()
        #     plotIntermediate(p_.ravel(), it=i, label=class_label, save=True, name="alignx")

        #     if class_ordering is True:
        #         range_limits = accumulate_move_force(p, range_limits, class_label, x_range)
        #         # print(range_limits[0, :])
            
        #     p[:, 0] = np.clip(p[:, 0], np.min(p[:, 1]) + (range_limits[:, 0] / 100) * x_range, np.min(p[:, 1]) + (range_limits[:, 1] / 100) * x_range)

        #     # plot after iter: ordered (optional) and moved
        #     p_ = p.copy()
        #     plotIntermediate(p_.ravel(), it=i, label=class_label, save=True, name="moved")

        #     p = p.ravel()

            # consistent with original y-axis size version
            # also not from 0 but from min_y
            # y_range = (np.max(p[:, 0]) - np.min(p[:, 0]))
            # p[:, 0] = np.clip(p[:, 0], np.min(p[:, 0]) + (range_limits[:, 0] / 100) * y_range, np.min(p[:, 0]) + (range_limits[:, 1] / 100) * y_range)
            # p = p.ravel()

            # accumulate force to move classes

        # clip (fix dimension) and re-order !!!every iter!!!
        if range_limits is not None:
            
            # print(range_limits[:, 0].shape[0])
            p = p.reshape(range_limits[:, 0].shape[0], 2)

            # plot and save every set iters
            if i != 0 and i % 50 == 0:
                p_ = p.copy()
                plotIntermediate(p_.ravel(), it=i, label=class_label, save=True, name="init")

            # resize as x-axis version
            current_range = np.max(p[:, 0]) - np.min(p[:, 0])
            x_range = (np.max(p[:, 1]) - np.min(p[:, 1]))
            scaling_factor = x_range / current_range if current_range != 0 else 1

            # rescale, also align with center at 0
            p[:, 0] = (p[:, 0] - np.mean(p[:, 0])) * scaling_factor

            # plot rescaled
            if i != 0 and i % 50 == 0:
                p_ = p.copy()
                plotIntermediate(p_.ravel(), it=i, label=class_label, save=True, name="alignx")

            if class_ordering is True:
                if i % 50 == 0:
                    print(f"iteration: {i}, force")
                    range_limits = accumulate_move_force(p, range_limits, class_label, x_range, out=True)
                else:
                    range_limits = accumulate_move_force(p, range_limits, class_label, x_range, out=False)
                # print(range_limits[0, :])
            
            # clip move mode
            if mode == "clip":
                p[:, 0] = np.clip(p[:, 0], np.min(p[:, 1]) + (range_limits[:, 0] / 100) * x_range, np.min(p[:, 1]) + (range_limits[:, 1] / 100) * x_range)
            elif mode == "uniform": # set +-5 as range
                p[:, 0] = np.clip(p[:, 0], np.min(p[:, 1]) + ((range_limits[:, 0] - 5) / 100) * x_range, np.min(p[:, 1]) + ((range_limits[:, 1] + 5) / 100) * x_range)
                range_limits = np.column_stack((p[:, 0], p[:, 0]))
                
            # plot after iter: ordered (optional) and moved
            if i != 0 and i % 50 == 0:
                p_ = p.copy()
                plotIntermediate(p_.ravel(), it=i, label=class_label, save=True, name="moved")

            p = p.ravel()

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc
            grad_norm = linalg.norm(grad)

            if verbose >= 2:
                print(
                    "[t-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: gradient norm %f. Finished."
                        % (i + 1, grad_norm)
                    )
                break

    return p, error, i


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "X_embedded": ["array-like", "sparse matrix"],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
    },
    prefer_skip_nested_validation=True,
)
def trustworthiness(X, X_embedded, *, n_neighbors=5, metric="euclidean"):
    # TODO: unused
    r"""Indicate to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
        (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : {array-like, sparse matrix} of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, default=5
        The number of neighbors that will be considered. Should be fewer than
        `n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
        mentioned in [1]_. An error will be raised otherwise.

    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, for a list
        of available metrics, see the documentation of argument metric in
        `sklearn.pairwise.pairwise_distances` and metrics listed in
        `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
        "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.

        .. versionadded:: 0.20

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.

    References
    ----------
    .. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
           Preservation in Nonlinear Projection Methods: An Experimental Study.
           In Proceedings of the International Conference on Artificial Neural Networks
           (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.

    .. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
           Local Structure. Proceedings of the Twelfth International Conference on
           Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.manifold import trustworthiness
    >>> X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
    >>> X_embedded = PCA(n_components=2).fit_transform(X)
    >>> print(f"{trustworthiness(X, X_embedded, n_neighbors=5):.2f}")
    0.92
    """
    n_samples = _num_samples(X)
    if n_neighbors >= n_samples / 2:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) should be less than n_samples / 2"
            f" ({n_samples / 2})"
        )
    dist_X = pairwise_distances(X, metric=metric)
    if metric == "precomputed":
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    ind_X_embedded = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(X_embedded)
        .kneighbors(return_distance=False)
    )

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
    ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
    )
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )
    return t

def plotIntermediate(p, it, label=None, show=False, save=False, name="default"):
    n_points = len(p) // 2
    p_reshaped = p.reshape(n_points, 2)  # Reshape to (n_points, 2)
    
    plt.figure(figsize=(8, 6))
    if label is not None:
        plt.scatter(p_reshaped[:, 1], p_reshaped[:, 0], c=label.astype(int), cmap='tab10', edgecolor='face', s=12)
        plt.colorbar()
    else:
        plt.scatter(p_reshaped[:, 1], p_reshaped[:, 0], edgecolor='face', s=12)
    plt.title(f"Intermediate iteration: {it}")
    if save:
        plt.savefig(f'.\\figures\\iter_{it}_{name}.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

class TSNEDimenfix(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "perplexity": [Interval(Real, 0, None, closed="neither")],
        "early_exaggeration": [Interval(Real, 1, None, closed="left")],
        "learning_rate": [
            StrOptions({"auto"}),
            Interval(Real, 0, None, closed="neither"),
        ],
        "max_iter": [Interval(Integral, 250, None, closed="left"), None],
        "n_iter_without_progress": [Interval(Integral, -1, None, closed="left")],
        "min_grad_norm": [Interval(Real, 0, None, closed="left")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "init": [
            StrOptions({"pca", "random"}),
            np.ndarray,
        ],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "method": [StrOptions({"barnes_hut", "exact"})],
        "angle": [Interval(Real, 0, 1, closed="both")],
        "n_jobs": [None, Integral],
        "n_iter": [
            Interval(Integral, 250, None, closed="left"),
            Hidden(StrOptions({"deprecated"})),
        ],
    }

    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_MAX_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=None,  # TODO(1.7): set to 1000
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        init="pca",
        verbose=0,
        random_state=None,
        method="barnes_hut",
        angle=0.5,
        n_jobs=None,
        n_iter="deprecated",
        range_limits=None,
        class_ordering=False,
        class_label=None,
        fix_iter=30,
        mode="clip",
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.range_limits = range_limits
        self.class_ordering = class_ordering
        self.class_label = class_label
        self.fix_iter = fix_iter
        self.mode = mode

    def _check_params_vs_input(self, X):
        if self.perplexity >= X.shape[0]:
            raise ValueError("perplexity must be less than n_samples")

    def _fit(self, X, skip_num_points=0):
        """Private function to fit the model using X as training data."""

        if isinstance(self.init, str) and self.init == "pca" and issparse(X):
            raise TypeError(
                "PCA initialization is currently not supported "
                "with the sparse input matrix. Use "
                'init="random" instead.'
            )

        if self.learning_rate == "auto":
            # See issue #18018
            self.learning_rate_ = X.shape[0] / self.early_exaggeration / 4
            self.learning_rate_ = np.maximum(self.learning_rate_, 50)
        else:
            self.learning_rate_ = self.learning_rate

        if self.method == "barnes_hut":
            X = validate_data(
                self,
                X,
                accept_sparse=["csr"],
                ensure_min_samples=2,
                dtype=[np.float32, np.float64],
            )
        else:
            X = validate_data(
                self,
                X,
                accept_sparse=["csr", "csc", "coo"],
                dtype=[np.float32, np.float64],
            )
        if self.metric == "precomputed":
            if isinstance(self.init, str) and self.init == "pca":
                raise ValueError(
                    'The parameter init="pca" cannot be used with metric="precomputed".'
                )
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(
                X,
                (
                    "TSNE.fit(). With metric='precomputed', X "
                    "should contain positive distances."
                ),
            )

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    "or provide the dense distance matrix."
                )

        if self.method == "barnes_hut" and self.n_components > 3:
            raise ValueError(
                "'n_components' should be inferior to 4 for the "
                "barnes_hut algorithm as it relies on "
                "quad-tree or oct-tree."
            )
        random_state = check_random_state(self.random_state)

        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    # Euclidean is squared here, rather than using **= 2,
                    # because euclidean_distances already calculates
                    # squared distances, and returns np.sqrt(dist) for
                    # squared=False.
                    # Also, Euclidean is slower for n_jobs>1, so don't set here
                    distances = pairwise_distances(X, metric=self.metric, squared=True)
                else:
                    metric_params_ = self.metric_params or {}
                    distances = pairwise_distances(
                        X, metric=self.metric, n_jobs=self.n_jobs, **metric_params_
                    )

            if np.any(distances < 0):
                raise ValueError(
                    "All distances should be positive, the metric given is not correct"
                )

            if self.metric != "euclidean":
                distances **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(
                P <= 1
            ), "All probabilities should be less or then equal to one"

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors...".format(n_neighbors))

            # Find the nearest neighbors for every point
            knn = NearestNeighbors(
                algorithm="auto",
                n_jobs=self.n_jobs,
                n_neighbors=n_neighbors,
                metric=self.metric,
                metric_params=self.metric_params,
            )
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print(
                    "[t-SNE] Indexed {} samples in {:.3f}s...".format(
                        n_samples, duration
                    )
                )

            t0 = time()
            distances_nn = knn.kneighbors_graph(mode="distance")
            duration = time() - t0
            if self.verbose:
                print(
                    "[t-SNE] Computed neighbors for {} samples in {:.3f}s...".format(
                        n_samples, duration
                    )
                )

            # Free the memory used by the ball_tree
            del knn

            # knn return the euclidean distance but we need it squared
            # to be consistent with the 'exact' method. Note that the
            # the method was derived using the euclidean method as in the
            # input space. Not sure of the implication of using a different
            # metric.
            distances_nn.data **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities_nn(distances_nn, self.perplexity, self.verbose)

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == "pca":
            pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=random_state,
            )
            # Always output a numpy array, no matter what is configured globally
            pca.set_output(transform="default")
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
            # the default value for random initialization. See issue #18018.
            X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
        elif self.init == "random":
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.standard_normal(
                size=(n_samples, self.n_components)
            ).astype(np.float32)

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(
            P,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            neighbors=neighbors_nn,
            skip_num_points=skip_num_points,
        )

    def _tsne(
        self,
        P,
        degrees_of_freedom,
        n_samples,
        X_embedded,
        neighbors=None,
        skip_num_points=0,
    ):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate_,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_MAX_ITER,
            "max_iter": self._EXPLORATION_MAX_ITER,
            "momentum": 0.5,
            "range_limits": self.range_limits,
            "class_ordering": self.class_ordering,
            "class_label": self.class_label,
            "fix_iter": self.fix_iter,
        }
        if self.method == "barnes_hut":
            obj_func = _kl_divergence_bh
            opt_args["kwargs"]["angle"] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args["kwargs"]["verbose"] = self.verbose
            # Get the number of threads for gradient computation here to
            # avoid recomputing it at each iteration.
            opt_args["kwargs"]["num_threads"] = _openmp_effective_n_threads()
        else:
            obj_func = _kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
        if self.verbose:
            print(
                "[t-SNE] KL divergence after %d iterations with early exaggeration: %f"
                % (it + 1, kl_divergence)
            )

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self._max_iter - self._EXPLORATION_MAX_ITER
        if it < self._EXPLORATION_MAX_ITER or remaining > 0:
            opt_args["max_iter"] = self._max_iter
            opt_args["it"] = it + 1
            opt_args["momentum"] = 0.8
            opt_args["n_iter_without_progress"] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print(
                "[t-SNE] KL divergence after %d iterations: %f"
                % (it + 1, kl_divergence)
            )

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    @_fit_context(
        # TSNE.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        # TODO(1.7): remove
        # Also make sure to change `max_iter` default back to 1000 and deprecate None
        if self.n_iter != "deprecated":
            if self.max_iter is not None:
                raise ValueError(
                    "Both 'n_iter' and 'max_iter' attributes were set. Attribute"
                    " 'n_iter' was deprecated in version 1.5 and will be removed in"
                    " 1.7. To avoid this error, only set the 'max_iter' attribute."
                )
            warnings.warn(
                (
                    "'n_iter' was renamed to 'max_iter' in version 1.5 and "
                    "will be removed in 1.7."
                ),
                FutureWarning,
            )
            self._max_iter = self.n_iter
        elif self.max_iter is None:
            self._max_iter = 1000
        else:
            self._max_iter = self.max_iter

        self._check_params_vs_input(X)
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    @_fit_context(
        # TSNE.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X)
        return self

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.embedding_.shape[1]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.metric == "precomputed"
        return tags
