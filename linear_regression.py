
# coding: utf-8
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import numpy as np

diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target
X, y = shuffle(data, target, random_state=13)
X = X.astype(np.float32)

offset = int(X.shape[0]*0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1, 1))

def linear_loss(X, y, w, b):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X, w)+b
    loss = np.sum((y_hat-y)**2)/num_train
    dw = np.dot(X.T, (y_hat-y))/num_train
    db = np.sum((y_hat-y))/num_train
    return y_hat, loss, dw, db

def initialize_params(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b

def linear_train(X, y, learning_rate, epochs):
    w, b = initialize_params(X.shape[1])
    loss_list = []
    for i in range(1, epochs):
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        loss_list.append(loss)
        w += -learning_rate*dw
        b += -learning_rate*db
        
        if i%1000 == 0:
            print('epoch %d loss %f' % (i, loss))
            
        params = {'w':w, 'b':b}
        grads = {'dw':dw, 'db':db}
    return loss_list, loss, params, grads

loss_list, loss, params, grads = linear_train(X_train, y_train, 0.001, 100000)

def predict(X, params):
    w = params['w']
    b = params['b']

    y_pred = np.dot(X, w) + b    
    return y_pred

y_pred = predict(X_test, params)
y_pred[:5]

