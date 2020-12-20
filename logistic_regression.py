
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
print(y_train.shape, y_test.shape)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def sigmoid(x):
    z = 1/(1+np.exp(-x))
    return z

def initialize_params(dims):
    W = np.zeros((dims, 1))
    b = 0
    return W, b

def logistic(X, y, W, b):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    
    a = sigmoid(np.dot(X, W)+b)
    cost = -1/num_train * np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    
    dW = np.dot(X.T, (a-y))/num_train
    db = np.sum(a-y)/num_train
    cost = np.squeeze(cost)
    
    return a, cost, dW, db

def logistic_train(X, y, learning_rate, epochs):
    W, b = initialize_params(X.shape[1])
    cost_list = []
    
    for i in range(epochs):
        a, cost, dW, db = logistic(X, y, W, b)
        W = W-learning_rate*dW
        b = b-learning_rate*db
        
        if i%10 == 0:
            cost_list.append(cost)
        if i%10 == 0:
            print('epoch %d cost %f' % (i, cost))
    
    params = {'W': W, 'b': b}
    grads = {'dW': dW, 'db': db}
    return cost_list, params, grads

def predict(X, params):
    y_prediction = sigmoid(np.dot(X, params['W']) + params['b'])
    for i in range(len(y_prediction)):
        if y_prediction[i]>0.5:
            y_prediction[i] = 1
        else:
            y_prediction[i] = 0
    return y_prediction

cost_list, params, grads = logistic_train(X_train, y_train, 0.00001, 100)
y_prediction = predict(X_test, params)
print(y_prediction)

def accuracy(y_test, y_pred):
    correct_count = 0
    for i in range(len(y_test)):        
        for j in range(len(y_pred)):            
            if y_test[i] == y_pred[j] and i == j:
                correct_count +=1

    accuracy_score = correct_count / len(y_test)    
    return accuracy_score
    
# 打印训练准确率
accuracy_score_train = accuracy(y_test, y_test_pred)
print(accuracy_score_train)

