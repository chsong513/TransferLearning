import numpy as np
class JointDistributionAdaptation(object):
    def __init__(self, source_X, source_Y, target_X, tartet_Y, lamda=1.0, gamma=1.0, kernel_type='rbf', iterations=10, Y_pseudo=None):
        self.source_X = source_X
        self.source_Y = source_Y
        self.target_X = target_X
        self.target_Y = tartet_Y
        self.lamda = lamda
        self.gamma = gamma
        self.kernel_type = kernel_type
        self.iterations = iterations
        self.Y_pseudo = Y_pseudo

    def fit(self):
        ns, nt = len(self.source_Y), len(self.target_Y)
        n = ns + nt

        X = np.

