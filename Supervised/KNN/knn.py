import numpy as np
from collections import Counter
"""
X refers to the training examples
y refers to the labels of the training examples (classes)
"""
class KNN_Classifier:
    def __init__(self, k):
        self.k = k #number of nearest neighbours

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self.__predict_helper(x) for x in X])

    #__ is used to make the function private
    def __predict_helper(self, x):
        #Euclidean distance
        distances = [np.sqrt(np.sum((x - x_train_sample)**2)) for x_train_sample in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices] #labels of k nearest neighbour
        return Counter(k_nearest_labels).most_common(1)[0][0] #most common label


class KNN_Regressor:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        return np.array([self.__predict_helper(x) for x in X])
    
    def __predict_helper(self, x):
        #Euclidean distance
        distances = [np.sqrt(np.sum((x - x_train_sample)**2)) for x_train_sample in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_values = self.y_train[k_indices] #labels of k nearest neighbor
        return np.mean(k_nearest_values) #average of k nearest neighbor values