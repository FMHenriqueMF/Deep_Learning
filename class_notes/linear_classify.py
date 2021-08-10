import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

np.random.seed(46)

X,Y = make_classification(n_features = 2,
                          n_redundant = 0,
                          n_informative = 1,
                          n_clusters_per_class = 1)


def plotmodel(w1,w2,b):
    plt.scatter(X[:,0],X[:,1], marker = 'o', c = Y)

    xmin,xmax = plt.gca().get_xlim()
    ymin,ymax = plt.gca().get_xlim()

    x = np.linspace(-2,4,50)
    y = (-w1*x-b)/w2

    plt.axvline(0,-1,1,color='k',linewidth=1)
    plt.axhline(0,-2,4,color='k',linewidth=1)
    plt.plot(x,y)
    plt.grid(True)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)


w1,w2,b = 5,1,-0.4
plotmodel(w1,w2,b)
plt.show()

def classify(ponto,w1,w2,b):
    ret = w1*ponto[0] +w2*ponto[1] +b
    if ret >= 0:
        return 1,'yellow'
    else:
        return 0, 'blue'

p = (2,-1)
classe,cor = classify(p,w1,w2,b)
print(classe,cor)
acertos = 0
for k in range(len(X)):
    categ, _ = classify(X[k],w1,w2,b)
    if categ == Y[k]:
        acertos += 1
print("acurácia: {0}%".format(100*acertos/len(X)))
