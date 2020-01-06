from transferlearning.traditional.jda import JointDistributionAdaptation
import numpy as np
import scipy.io as scio
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

usps = scio.loadmat('data/usps.mat.bin')
mnist = scio.loadmat('data/mnist.mat.bin')

domain_names = ['mnist', 'usps']
domains = [mnist, usps]

def mykernel(X, Y=None):
    if Y is not None:
        K = np.dot(X.T, Y)
    else:
        K = np.dot(X.T, X)
    return K

def main():
    for i in range(len(domains)):
        for j in range(len(domains)):
            if i != j:
                print('\n from', domain_names[i], 'to', domain_names[j])

                source, target = domains[i], domains[j]
                baseline = KNeighborsClassifier(n_neighbors=1)
                baseline.fit(source['X'].T, source['Y'].flatten())
                Y_pseudo_target = baseline.predict(target['X'].T)
                baseline_acc = sklearn.metrics.accuracy_score(target['Y'].flatten(), Y_pseudo_target)
                print('    acc of baseline 1-NN:', baseline_acc)

                jda = JointDistributionAdaptation(source['X'], source['Y'], target['X'], target['Y'],
                                                  classifier=KNeighborsClassifier(n_neighbors=1), iterations=1,
                                                  Y_pseudo=Y_pseudo_target)
                X_JDA_source, X_JDA_target = jda.adapt()
                jda = KNeighborsClassifier(n_neighbors=1)
                jda.fit(X_JDA_source.T, source['Y'].flatten())
                Y_pseudo_target = jda.predict(X_JDA_target.T)
                jda_acc = sklearn.metrics.accuracy_score(target['Y'].flatten(), Y_pseudo_target)
                print('    acc of jda:', jda_acc)

if __name__ == '__main__':
    main()

