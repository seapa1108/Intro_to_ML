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


# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {} # P(Class)
        self.likelihoods = {} # P(Feature|Class)

    def calculate_prior(self, y): # P(Class)
        self.priors[0] = 1 - np.sum(y) / len(y)
        self.priors[1] = np.sum(y) / len(y)

    def calculate_likelihood(self, X, y): # P(Feature|Class)
        y = y.squeeze()
        num_features = X.shape[1]
        self.likelihoods = {0: {}, 1: {}}

        for feature_idx in range(num_features):
            if len(np.unique(X[:, feature_idx])) == 2: # Binary
                for cls in [0, 1]:
                    cls_samples = X[y == cls, feature_idx]
                    alpha = 1
                    self.likelihoods[cls][feature_idx] = {
                        "type": "binary",
                        0: (np.sum(cls_samples == 0) + alpha) / (len(cls_samples) + (alpha * 2)),
                        1: (np.sum(cls_samples == 1) + alpha) / (len(cls_samples) + (alpha * 2)),
                    }
                
            else: # Continuous
                for cls in [0, 1]:
                    cls_samples = X[y == cls, feature_idx]
                    mean = np.mean(cls_samples)
                    variance = np.var(cls_samples) + 1e-9
                    self.likelihoods[cls][feature_idx] = {"type": "continuous", "mean": mean, "variance": variance}

    def calculate_prob(self, x, cls):
        prob = np.log(self.priors[cls])

        for feature_idx, feature_value in enumerate(x):
            if self.likelihoods[cls][feature_idx]["type"] == "binary":
                prob += np.log(self.likelihoods[cls][feature_idx][feature_value])
            else:
                mean = self.likelihoods[cls][feature_idx]["mean"]
                variance = self.likelihoods[cls][feature_idx]["variance"]
                prob += np.log(self.gaussian_pdf(feature_value, mean, variance))

        return prob              

    def gaussian_pdf(self, x, mean, variance):
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

    def fit(self, X, y):
        self.calculate_prior(y)
        self.calculate_likelihood(X, y)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(0 if self.calculate_prob(x, 0) > self.calculate_prob(x, 1) else 1)
        return np.array(predictions)

    def predict_proba(self, X):
        probabilities = []
        for x in X:
            posteriors = [np.exp(self.calculate_prob(x, 0)), np.exp(self.calculate_prob(x, 1))]
            total = sum(posteriors)
            probabilities.append([posteriors / total])
        return np.array(probabilities)