# Transfer Learning
[![Build Status](https://travis-ci.com/chsong513/TransferLearning.svg?branch=master)](https://travis-ci.com/chsong513/TransferLearning)
[![PyPI](https://img.shields.io/pypi/v/transferlearning)](https://pypi.org/project/transferlearning/)
[![Language](https://img.shields.io/badge/language-python-green.svg)](https://pypi.org/project/transferlearning/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/transferlearning)](https://pypi.org/project/transferlearning/)
[![PyPI - License](https://img.shields.io/pypi/l/transferlearning)](https://github.com/chsong513/TransferLearning)

Integrate some classic transfer learning algorithms and methods, the non-deep methods will be implemented by python3, 
the deep methods will be implemented with pytorch. 
Here, the first algorithm Joint Distribution Adaptation (JDA) has been implemented.

## Documents
The documents will be released later. 

## Quick Start
### Setup
```
pip install --upgrade transferlearning
```
If you meet some thing wrong like "ERROR: Could not install packages due to an EnvironmentError: HTTPSConnectionPool]", you can create a folder "C:\Users\Administrator\pip" and creat a file "pip.ini" under this folder, then, copy the next content into the file. Finally, try to install this package again.

```
[global]

trusted-host=mirrors.aliyun.com

index-url=http://mirrors.aliyun.com/pypi/simple/
```
### Utilizing
```python
import transferlearning as tl
```

### Example
An example of this transferlearning of joint distribution adaptation.  

#### Import packages
```python
import transferlearning as tl
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as scio
import sklearn.metrics
import numpy as np
```

#### Acquire Data, Source and target domains
The data used in this example are two digit datasets: mnist and usps.
They can be downloaded from [TianChi](https://tianchi.aliyun.com/dataset/dataDetail?dataId=48612).
```python
usps = scio.loadmat('data/usps.mat.bin')
mnist = scio.loadmat('data/mnist.mat.bin')
source, target = mnist, usps
```

#### Baseline
Adopt 1-NN as the baseline, train it on source data, and test it on target data.
```python
baseline = KNeighborsClassifier(n_neighbors=1)
'''train on source data'''
baseline.fit(source['X'].T, source['Y'].flatten())
'''test on target data'''
Y_pseudo_target = baseline.predict(target['X'].T)
baseline_acc = sklearn.metrics.accuracy_score(target['Y'].flatten(), Y_pseudo_target)
print('acc of baseline 1-NN:', baseline_acc)
```

#### JDA algorithm
The function `JointDistributionAdaptation` implement the JDA algorithm, it needs some parameters: 
- source_X : np.array of shape [num_of_features, num_of_instance]

- source_Y : np.array of shape [num_of_instance, 1]

- target_X : np.array of shape [num_of_features, num_of_instance]

- target_Y : np.array of shape [num_of_instance, 1]

- lamda : float, optional (default=1.0)

- gamma : float, optional (default=1.0)

- kernel : {'primal', 'linear', 'rbf', func}, optional (default='rbf')
    you can define kernel function for yourself, you only need to set kernel = mykernel
- iterations : int, optional (default=1)

- Y_pseudo : np.array of shape (num_of_instance, ), if Y_pseudo is None, JDA will be TCA

- classifier : func, optional (default=KNeighborsClassifier(n_neighbors=1))
    the classifier that used to generate pseudo label for target domain during training
    
```python
'''define your own kernel function'''
def mykernel(X, Y=None):
    if Y is not None:
        K = np.dot(X.T, Y)
    else:
        K = np.dot(X.T, X)
    return K
jda = tl.JointDistributionAdaptation(source['X'], source['Y'], target['X'], target['Y'], kernel=mykernel)
X_JDA_source, X_JDA_target = jda.adapt()
```

#### validate the effectiveness of JDA
Adopt another 1-NN classifier, train it on the JDA-processed source data, and test it on the JDA-processed target data.
```python
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_JDA_source.T, source['Y'].flatten())
Y_pseudo_target = clf.predict(X_JDA_target.T)
jda_acc = sklearn.metrics.accuracy_score(target['Y'].flatten(), Y_pseudo_target)
print('acc of jda:', jda_acc)
```

#### All Code
```python
import transferlearning as tl
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as scio
import sklearn.metrics
import numpy as np


usps = scio.loadmat('data/usps.mat.bin')
mnist = scio.loadmat('data/mnist.mat.bin')

def mykernel(X, Y=None):
    if Y is not None:
        K = np.dot(X.T, Y)
    else:
        K = np.dot(X.T, X)
    return K

def main():
    source, target = mnist, usps
    baseline = KNeighborsClassifier(n_neighbors=1)
    baseline.fit(source['X'].T, source['Y'].flatten())
    Y_pseudo_target = baseline.predict(target['X'].T)
    baseline_acc = sklearn.metrics.accuracy_score(target['Y'].flatten(), Y_pseudo_target)
    print('acc of baseline 1-NN:', baseline_acc)

    jda = tl.JointDistributionAdaptation(source['X'], source['Y'], target['X'], target['Y'], classifier=KNeighborsClassifier(n_neighbors=1), iterations=1, Y_pseudo=Y_pseudo_target)
    X_JDA_source, X_JDA_target = jda.adapt()
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_JDA_source.T, source['Y'].flatten())
    Y_pseudo_target = clf.predict(X_JDA_target.T)
    jda_acc = sklearn.metrics.accuracy_score(target['Y'].flatten(), Y_pseudo_target)
    print('acc of jda:', jda_acc)

if __name__ == '__main__':
    main()
```

#### Experimental Results
The result is great, even greater than the result in the [paper](http://openaccess.thecvf.com/content_iccv_2013/html/Long_Transfer_Feature_Learning_2013_ICCV_paper.html).
```
acc of baseline 1-NN: 0.6444444444444445
 iteration: 1 
  acc: 0.7461111111111111
acc of jda: 0.7461111111111111
```

## About Me

Song Cheng

NWPU SE Bachelor; USTC CS Master

Email: chsong513@gmail.com
