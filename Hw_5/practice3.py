import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# from google.colab import drive
# drive.mount('/content/gdrive/', force_remount=True)

# data_path = '/content/gdrive/MyDrive/DataScience7/data/'
data_path = '/content/gdrive/MyDrive/DataScience7/data/'

def plot_data(X, y, pos_label='y=1', neg_label='y=0'):
    positive = y == 1
    negative = y == 0

    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'k+', label=neg_label)

# Plot Boundary
def visualize_boundary(clf, X, x_min, x_max, y_min, y_max):
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='r')

# Part1: Loading and Visualizing Data
data = scio.loadmat(data_path + 'ex6data1.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

plot_data(X, y)

# SVM метод опорних векторів (create clasificator)
c = 1000  # low mistakes
clf = svm.SVC(C = c, kernel='linear')
clf.fit(X, y)

plt.figure(figsize = (12, 8))
plt.grid()
plot_data(X, y)
visualize_boundary(clf, X, 0, 4.5, 1.5, 5)

# RBF
data = scio.loadmat(data_path + 'ex6data2.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

plot_data(X, y)

c = 1  # let mistakes
sigma = 0.1

clf = svm.SVC(C = c, kernel='rbf', gamma=np.power(sigma, -2))
clf.fit(X, y)

plt.figure(figsize = (12, 8))
plt.grid()
plot_data(X, y)
visualize_boundary(clf, X, 0, 1, .4, 1.0)
