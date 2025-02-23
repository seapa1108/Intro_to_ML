import numpy as np
from sklearn.model_selection import KFold
from .base_learner import LogisticRegressionClassifier, MLPClassifier,   \
                DecisionTreeClassifier, KNearestNeighborClassifier,     \
                NaiveBayesClassifier
from tqdm import tqdm


class StackingClassifier:
    def __init__(self):
        """
        Stacking Classifier
        """
        self.LR = LogisticRegressionClassifier(penalty='l2', lr=0.005, iterations=8000)
        self.MLP = MLPClassifier(layers = [64, 16, 8, 1], activate_function = "sigmoid", 
                                 optimizer = "adam", learning_rate = 0.005, n_epoch = 7000)
        self.DT = DecisionTreeClassifier(max_depth=6)
        self.KNN = KNearestNeighborClassifier(k=33, distance_metric="manhattan")
        self.NB = NaiveBayesClassifier()
        self.META = NaiveBayesClassifier()
        self.base_learners = [self.LR, self.MLP, self.DT, self.KNN, self.NB]

    def fit(self, X_train, y_train):
        """
        Fit the stacking model.
        :param X_train: Training data.
        :param y_train: Training labels.
        """

        meta_features = np.zeros((len(X_train), len(self.base_learners)))

        num_folds = 10
        kf = KFold(n_splits=num_folds, random_state=42, shuffle=True)

        for i, model in enumerate(self.base_learners):
            for train_idx, val_idx in kf.split(X_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t = y_train[train_idx]

                model.fit(X_t, y_t)
                
                meta_features[val_idx, i] = model.predict(X_v)

        self.META.fit(meta_features, y_train)

        for model in self.base_learners:
            model.fit(X_train, y_train)
        

    def predict(self, X_test):
        """
        Predict using the stacking model.
        :param X_test: Test data.
        :return: Final predictions.
        """
        meta_features = []
        for model in self.base_learners:
            meta_features.append(model.predict(X_test))

        meta_features = np.column_stack(meta_features)

        final_predictions = self.META.predict(meta_features)
        return final_predictions
        