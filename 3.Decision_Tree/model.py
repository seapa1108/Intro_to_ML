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


class DecisionTreeClassifier:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None
        self.leaf_size_threshold = 5
        self.gain_threshold = 0.02

    def fit(self, X, y):
        self.leaf_size_threshold = int(len(X) / 50)
        self.tree = self._grow_tree(X, y)
        self._post_prune(self.tree, X, y)

    def _grow_tree(self, X, y, depth=0, used_feature=None):
        if used_feature is None:
            used_feature = []
        
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.leaf_size_threshold:
            if np.sum(y) > len(y) - np.sum(y):
                leaf_value = 1
            else:
                leaf_value = 0
            return {"is_leaf": True, "value": leaf_value}
        
        best_split_feature, best_split_threshold = self.find_best_split(X, y, used_feature)
        if best_split_feature is None or best_split_threshold is None:
            if np.sum(y) > len(y) - np.sum(y):
                leaf_value = 1
            else:
                leaf_value = 0
            return {"is_leaf": True, "value": leaf_value}
        
        used_feature.append(best_split_feature)

        X_left, X_right, y_left, y_right = self.split_dataset(X, y, best_split_feature, best_split_threshold)
        left_tree = self._grow_tree(X_left, y_left, depth + 1, used_feature)
        right_tree = self._grow_tree(X_right, y_right, depth + 1, used_feature)
        
        zeroes = len(y) - np.sum(y)
        ones = np.sum(y)

        return {
            "is_leaf": False,
            "feature_index": best_split_feature,
            "threshold": best_split_threshold,
            "left": left_tree,
            "right": right_tree,
            "zeroes": zeroes,
            "ones": ones,
            "left_size": len(y_left),
            "right_size": len(y_right)
        } 

    # Split dataset based on a feature and threshold
    def split_dataset(self, X, y, feature_index, threshold):
        left_idx = np.where(X[:, feature_index] <= threshold)
        right_idx = np.where(X[:, feature_index] > threshold)
            
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    # Find the best split for the dataset
    def find_best_split(self, X, y, used_features):
        best_split = [None, None] # feature_idx, threshold
        max_gain = self.gain_threshold

        parent_entropy = self.entropy(y)
        for feature_idx in range(X.shape[1]):            
            if feature_idx in used_features:
                continue

            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split_dataset(X, y, feature_idx, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                left_entropy = self.entropy(y_left)
                right_entropy = self.entropy(y_right)
                gain = parent_entropy - (len(y_left) / len(y) * left_entropy + len(y_right) / len(y) * right_entropy)
                if gain > max_gain:
                    max_gain = gain
                    best_split = [feature_idx, threshold]
        return best_split
    

    def entropy(self, y):
        entropy = 0
        labels = np.unique(y)
        for label in labels:
            count = np.sum(y == label)
            p = count / len(y)
            entropy += (-1) * p * np.log2(p)

        return entropy
    
    # prediction
    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        if tree_node["is_leaf"]:
            return tree_node["value"]
        
        next = tree_node["left"] if x[tree_node["feature_index"]] <= tree_node["threshold"] else tree_node["right"]
        
        return self._predict_tree(x, next)
    
    # print tree
    def print_tree(self, max_print_depth=3, node = None, depth = 0):
        if node is None:
            node = self.tree
        if depth >= max_print_depth:
            return
        
        if not node["is_leaf"]:
            print("  " * depth + f'[F{node["feature_index"]}] [{node["zeroes"]} 0 / {node["ones"]} 1] [{node["left_size"]} left / {node["right_size"]} right]')
            if not (node["left"]["is_leaf"] or depth + 1 >= max_print_depth):
                print("  " * depth + "Left:")
            self.print_tree(max_print_depth, node=node["left"], depth=depth + 1)
            if not (node["right"]["is_leaf"] or depth + 1 >= max_print_depth):
                print("  " * depth + "Right:")
            self.print_tree(max_print_depth, node=node["right"], depth=depth + 1)

    def _post_prune(self, node, X, y):
        if node["is_leaf"]:
            return
        
        if "left" in node and not node["left"]["is_leaf"]:
            self._post_prune(node["left"], X[X[:, node["feature_index"]] <= node["threshold"]], y[X[:, node["feature_index"]] <= node["threshold"]])
        if "right" in node and not node["right"]["is_leaf"]:
            self._post_prune(node["right"], X[X[:, node["feature_index"]] > node["threshold"]], y[X[:, node["feature_index"]] > node["threshold"]])

        if node["left"]["is_leaf"] and node["right"]["is_leaf"]:
            left_leaf_value = node["left"]["value"]
            right_leaf_value = node["right"]["value"]

            y_pred = np.where(X[:, node["feature_index"]] <= node["threshold"], left_leaf_value, right_leaf_value)
            split_error = np.sum(y_pred != y.flatten()) / len(y_pred)

            majority = 1 if np.sum(y) > len(y) - np.sum(y) else 0
            prune_error = np.sum(majority != y.flatten()) / len(y.flatten())
            
            if prune_error < split_error:
                node["is_leaf"] = True
                node["value"] = majority
                del node["left"]
                del node["right"]