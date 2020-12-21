
# coding: utf-8
import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

iris=load_iris()
X, y = shuffle(iris.data, iris.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0]*0.7)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

print('X_train=', X_train.shape)
print('X_test=', X_test.shape)
print('y_train=', y_train.shape)
print('y_test=', y_test.shape)

def compute_distances(X, X_train):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    M = np.dot(X, X_train.T)
    te = np.square(X).sum(axis=1)
    tr = np.square(X_train).sum(axis=1)
    dists = np.sqrt(-2*M + tr + np.matrix(te).T)
    return dists

def predict_labels(y_train, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = []
        labels = y_train[np.argsort(dists[i,:])].flatten()
        closest_y = labels[0:k]
        
        c = Counter(closest_y)
        y_pred[i] = c.most_common(1)[0][0]
    return y_pred

def cross_validation(X_train, y_train):
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    
    X_train_folds = []
    y_train_folds = []
    
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    
    k_to_accuracies = {}
    for k in k_choices:
        for fold in range(num_folds):
            validation_X_test = X_train_folds[fold]
            validation_y_test = y_train_folds[fold]
            temp_X_train = np.concatenate(X_train_folds[:fold]+X_train_folds[fold+1:])
            temp_y_train = np.concatenate(y_train_folds[:fold]+y_train_folds[fold+1:])
            
            temp_dists = compute_distances(validation_X_test, X_train)
            temp_y_test_pred = predict_labels(y_train, temp_dists, k=k)
            temp_y_test_pred = temp_y_test_pred.reshape((-1, 1))
            num_correct = np.sum(temp_y_test_pred == validation_y_test)
            num_test = validation_X_test.shape[0]
            accuracy = float(num_correct)/num_test
            k_to_accuracies[k] = k_to_accuracies.get(k, [])+[accuracy]
            
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy=%f' % (k, accuracy))
            
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    best_k = k_choices[np.argmax(accuracies_mean)]
    print('最佳k值为{}'.format(best_k))
    
    return best_k

if __name__ == '__main__':
    best_k = cross_validation(X_train, y_train)
    dists = compute_distances(X_test, X_train)
    y_test_pred = predict_labels(y_train, dists, k=best_k)
    y_test_pred = y_test_pred.reshape((-1, 1))
    num_correct = np.sum(y_test_pred==y_test)
    accuracy = float(num_correct)/X_test.shape[0]
    print('Got %d / %d correct => accuracy %f' % (num_correct, X_test.shape[0], accuracy))
