import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

torch.manual_seed(199)
perceptron = nn.Linear(3,1)

w1,w2,w3 = perceptron.weight.data.numpy()[0]
b = perceptron.bias.data.numpy()


def plot3d(perceptron):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    w1,w2,w3 = perceptron.weight.data.numpy()[0]
    b = perceptron.bias.data.numpy()
    X1 = np.linspace(-1,1,10)
    X2 = np.linspace(-1,1,10)

    X1,X2 = np.meshgrid(X1,X2)
    X3 = (b - w1*X1 - w2*X2)/w3

    surf = ax.plot_surface(X1,X2,X3,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)



X = torch.Tensor([0,-1,2])
y = perceptron(X)
print(y)
plot3d(perceptron)
plt.plot([X[0]],[X[1]],[X[2]],marker='^',markersize=20)
plt.show()
