%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from warnings import filterwarnings
filterwarnings('ignore')

data = np.loadtxt('data.csv', delimiter=',')
X = data[:, :2]
y = data[:, 2]
label_mask = np.equal(y, 1)

plt.scatter(X[:, 0][label_mask], X[:, 1][label_mask], color='red')
plt.scatter(X[:, 0][~label_mask], X[:, 1][~label_mask], color='blue')
plt.show()


#Train data using LogisticRegression classes from skikit-learn library.
def learn_and_return_weights(X, y):
    from sklearn.linear_model import LogisticRegression
    # YOUR CODE COMES HERE
    # w: coefficient of the model to input features,
    # b: bias of the model
    model = LogisticRegression(solver="liblinear")
    model = model.fit(X, y)
    w = model.coef_[0]
    b = model.intercept_
    return w, b
  
  def plot_data_and_weights(X, y, w, b):
    plt.scatter(X[:, 0][label_mask], X[:, 1][label_mask], color='red')
    plt.scatter(X[:, 0][~label_mask], X[:, 1][~label_mask], color='blue')

    x_lin = np.arange(20, 70)
    y_lin = -(0.5 + b + w[0] * x_lin) / w[1]

    plt.plot(x_lin, y_lin, color='black');
    plt.show()

w, b = learn_and_return_weights(X, y)
plot_data_and_weights(X, y, w, b)

#Implement Logistic Regression without using scikit-learn libraries.
def sigmoid(z):
    # YOUR CODE COMES HERE
    return 1/(1 + np.exp(-z))
     

def binary_cross_entropy_loss(y_pred, target):
    # YOUR CODE COMES HERE
    cross=0
    for i in range(y_pred.shape[0]):
      cross += np.sum(y_pred[i] * np.log(target[i])) + (1 - y_pred[i]) * np.log(1 - target[i])
    return cross/y_pred.shape[0]

def learn_and_return_weights_numpy(X, Y, lr=.01, iter=100000):
    # YOUR CODE COMES HERE
    # w: coefficient of the model to input features,
    # b: bias of the model
    w = np.zeros(X.shape[1])
    b = 0.1
    for i in range(iter):
      z = np.dot(X,w) + b
      y_pred = sigmoid(z)
      l = binary_cross_entropy_loss(y_pred, Y)
      dw = np.dot((y_pred-Y).T, X)/X.shape[0]
      db = np.mean(y_pred-Y)
      w = w - lr*dw
      b = b - lr*db
    return w, b
  
w, b = learn_and_return_weights_numpy(X, y)
plot_data_and_weights(X, y, w, b)
