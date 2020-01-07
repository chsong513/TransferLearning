# coding: utf-8
# Joint Distribution Adaptation
# 2019/12/27 @ chengsong

import numpy as np
from transferlearning.nn.utils import kernel
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

class JointDistributionAdaptation(object):
    """
    Parameters
    ----------
    source_X : np.array of shape [num_of_features, num_of_instance]

    source_Y : np.array of shape [num_of_instance, 1]

    target_X : np.array of shape [num_of_features, num_of_instance]

    target_Y : np.array of shape [num_of_instance, 1]

    lamda : float, optional (default=1.0)

    gamma : float, optional (default=1.0)

    kernel : {'primal', 'linear', 'rbf', func}, optional (default='rbf')

    iterations : int, optional (default=1)

    Y_pseudo : np.array of shape (num_of_instance, )

    classifier : func, optional (default=KNeighborsClassifier(n_neighbors=1))
        the classifier that used to generate pseudo label for target domain during training

    """
    def __init__(self, source_X, source_Y, target_X, tartet_Y, **kwargs):
        self.lamda = kwargs['lamda'] if 'lamda' in kwargs.keys() else 1.0
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs.keys() else 1.0
        self.mykernel = kwargs['kernel'] if 'kernel' in kwargs.keys() else 'rbf'
        self.iterations = kwargs['iterations'] if 'iterations' in kwargs.keys() else 1
        self.Y_pseudo = kwargs['Y_pseudo'] if 'Y_pseudo' in kwargs.keys() else None
        self.classifier = kwargs['classifier'] if 'classifier' in kwargs.keys() else KNeighborsClassifier(n_neighbors=1)
        self.source_X = source_X
        self.source_Y = source_Y
        self.target_X = target_X
        self.target_Y = tartet_Y

    def adapt(self):
        """
        Return
        ------
        X_source_JDA : np.array of shape [reduced_dimension_of_features, num_of_instance]

        X_target_JDA : np.array of shape [reduced_dimension_of_features, num_of_instance]
        """
        ns, nt = len(self.source_Y), len(self.target_Y)
        n = ns + nt

        X = np.concatenate((self.source_X, self.target_X), axis=1)
        X /= np.linalg.norm(X, axis=0)

        H = np.eye(n) - (1/n) * np.ones((n, n))

        e0 = np.concatenate((np.ones((ns, 1))/ns, -np.ones((nt, 1))/nt), axis=0)
        M0 = e0.dot(e0.T)

        for i in range(self.iterations):
            print(' iteration:', i+1, ' ')
            N = 0
            if self.Y_pseudo is not None:
                C = len(np.unique(self.source_Y))
                for c in range(1, C+1):
                    e = np.zeros((n, 1))
                    e[np.where(self.source_Y==c)] = 1/len(self.source_Y[np.where(self.source_Y==c)])
                    e[np.where(self.Y_pseudo == c)[0] + ns] = -1 / len(self.Y_pseudo[np.where(self.Y_pseudo == c)])
                    N += e.dot(e.T)
            M = M0 +N
            M /= np.linalg.norm(M, 'fro')

            if type(self.mykernel) is str:
                n_eye = X.shape[0] if self.mykernel == 'original' else n
                K = kernel(self.mykernel, X=X, gamma=self.gamma)
            else:
                K = self.mykernel(X)
                n_eye = n

            eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(np.linalg.multi_dot([K, H, K.T])).dot(
                np.linalg.multi_dot([K, M, K.T]) + self.lamda * np.eye(n_eye)))
            ind = np.argsort(eig_vals)
            A = eig_vecs[:, ind[:100]]
            Z = A.T.dot(K).astype(float)
            Z /= np.linalg.norm(Z, axis=0)

            self.X_source_JDA, self.X_target_JDA = Z[:,:ns], Z[:,ns:]
            self.classifier.fit(self.X_source_JDA.T, self.source_Y.flatten())
            Y_pseudo = self.classifier.predict(self.X_target_JDA.T)
            pseudo_acc = sklearn.metrics.accuracy_score(self.target_Y.flatten(), Y_pseudo)
            print(' acc:', pseudo_acc)
        return self.X_source_JDA, self.X_target_JDA