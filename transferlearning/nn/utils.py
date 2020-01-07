# coding: utf-8
# Kernel function
# 2019/12/27 @ chengsong

import numpy as np

def kernel(kernel_type, X, Y=None, gamma=1.0):
    """
    Parameters
    ----------
    kernel_type : {'primal', 'linear', 'rbf'}
        the type of the kernel:
        - 'primal':
        - 'linear':
        - 'rbf':

    X : np.array of shape [num_of_features, num_of_instance]

    Y : np.array of shape [num_of_features, num_of_instance], optional (default=None)

    gamma : float, optional (default=1.0)

    Returns
    -------
    K : np.array of shape [num_of_instance, num_of_instance]
        the Key matrix of X and Y

    Example
    -------
        X = np.random.randn(2,2)
        [[ 0.56274911  0.17874628]
        [ 0.0516268  -1.34844311]]

        Y = np.random.randn(2,3)
        [[-1.20692944  0.26468792 -0.6348792 ]
        [-0.96157471  1.10384177 -0.32264376]]

        K = kernel('rbf', X, Y)
        [[0.0156335  0.30240237 0.20713345]
        [0.12621552 0.00242703 0.18009862]]
    """

    if kernel_type == 'primal':
        K = X
    elif kernel_type == 'linear':
        if Y is not None:
            K = np.dot(X.T, Y)
        else:
            K = np.dot(X.T, X)
    elif kernel_type == 'rbf':
        Xsq = np.sum(X.T ** 2, axis=1).reshape(X.shape[1], -1)
        if Y is None:
            D = Xsq.dot(np.ones((1, X.shape[1]))) + np.ones((X.shape[1], 1)).dot(Xsq.T) - 2*X.T.dot(X)
        else:
            Ysq = np.sum(Y.T ** 2, axis=1).reshape(Y.shape[1], -1)
            D = Xsq.dot(np.ones((1, Y.shape[1]))) + np.ones((X.shape[1], 1)).dot(Ysq.T) - 2*X.T.dot(Y)
        K = np.exp(-gamma * D)
    else:
        raise Exception("Invalid kernel type!", kernel_type)
    return K
	
