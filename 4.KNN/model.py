import numpy as np
from abc import ABC, abstractmethod


# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

    @abstractmethod
    def predict_proba(self, X):
        # Abstract method predict the probability of the dataset X
        pass

# K-Nearest Neighbors Classifier
class KNearestNeighborClassifier(Classifier):
    def __init__(self, k=3, distance_metric='euclidean', p=3): 
        self.k = k
        self.p = p
        self.dist = distance_metric
        self.train_X = None
        self.train_y = None

    def fit(self, X, y): 
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X] 
        return np.array(y_pred)
    
    def predict_proba(self, X):
        #TODO
        pass

    def _predict(self, x):
        neighbors = self.neighbors(x)[:self.k]
        if np.sum(neighbors) > len(neighbors) * (116/426):
            return 1
        else:
            return 0

        
    def distance(self, x1, x2):
        if self.dist == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.dist == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.dist == 'chebyshev':
            return np.max(np.abs(x1 - x2))
        else: #Minkowski
            return (np.sum(np.abs(x1 - x2)**self.p)) ** (1/self.p)
    
    def neighbors(self, x):
        distances = np.array([self.distance(x, x2) for x2 in self.train_X])
        indices = np.argsort(distances)
        return self.train_y[indices]