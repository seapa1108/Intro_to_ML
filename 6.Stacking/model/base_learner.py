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



# Logistic Regression Classifier
class LogisticRegressionClassifier(Classifier):
    def __init__(self, C=1.0, penalty='l2', lr=1e-2, iterations=600):
        super(LogisticRegressionClassifier, self).__init__()

        self.iterations = iterations
        self.lr= lr
        self.C = C
        self.penalty = penalty

    def sigmoid(self, x):
        """ The sigmoid function """
        return 1.0 / (1.0 + np.exp(-x))
    
    def linear(self, X):
        return np.dot(X, self.W) + self.b
    
    def SGD(self, dW, db):
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

    def binaryCrossEntropy(self, pred, target):
        return -np.mean(target * np.log(pred + 1e-8) + (1 - target) * np.log(1 - pred + 1e-8))

    def fit(self, X, y):
        m, n = X.shape # (m, n)
        self.W = np.random.randn(n) # (n,)
        # self.W = np.zeros(n)
        self.b = 0
        self.loss = []

        y = np.array(y).flatten()  # make sure y is one dimension

        for i in range(self.iterations):
            z = self.linear(X)
            y_hat = self.sigmoid(z)

            dW = 1/m * np.dot(X.T, (y_hat - y))
            db = 1/m * np.sum(y_hat - y)

            if self.penalty == 'l2':
                dW += (self.C / m) * self.W
            elif self.penalty == 'l1':
                dW += (self.C / m) * np.sign(self.W)

            # update weight and bias
            self.SGD(dW, db)

            # compute loss
            loss = self.binaryCrossEntropy(y_hat, y)
            self.loss.append(loss)
            # if i % 50 == 0:
            #     print(f"Epoch {i}, Loss: {loss}")
    
    def predict(self, X):
        y_hat = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_hat]
    
    def predict_proba(self, X):
        z = self.linear(X)
        return self.sigmoid(z)
    

# ====== Activation funtion ====== #
class activation():
    def __init__(self):
        # TODO
        pass

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
        
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2  
    

# ====== Optimizer function ====== #
class optimizer():
    def __init__(self):
        # TODO
        pass

    def SGD(self, lr, W, dW, b, db):
        W -= lr * dW
        b -= lr * db
        return W, b
    
    def adagrad(self, lr, W, dW, b, db, prev_W, prev_b):
        prev_W += dW ** 2
        prev_b += db ** 2

        W -= lr * dW / (np.sqrt(prev_W) + 1e-7)
        b -= lr * db / (np.sqrt(prev_b) + 1e-7)

        return W, b, prev_W, prev_b

    def adam(self, lr, W, dW, b, db, t, s_W, s_b, r_W, r_b):
        s_W = 0.9 * s_W + (1 - 0.9) * dW
        s_b = 0.9 * s_b + (1 - 0.9) * db

        r_W = 0.999 * r_W + (1 - 0.999) * (dW ** 2)
        r_b = 0.999 * r_b + (1 - 0.999) * (db ** 2)

        s_W_hat = s_W / (1 - 0.9 ** t)
        s_b_hat = s_b / (1 - 0.9 ** t)

        r_W_hat = r_W / (1 - 0.999 ** t)
        r_b_hat = r_b / (1 - 0.999 ** t)

        W -= lr * s_W_hat / (np.sqrt(r_W_hat) + 1e-8)
        b -= lr * s_b_hat / (np.sqrt(r_b_hat) + 1e-8)

        return W, b, s_W, s_b, r_W, r_b


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



# MLP Classifier
class MLPClassifier(Classifier):
    def __init__(self, layers = [128, 8, 1], activate_function = "sigmoid", optimizer = "SGD", learning_rate = 0.005, n_epoch = 5000):
        """ TODO, Initialize your own MLP class """
        self.layers = layers

        Activation = activation()
        if activate_function == 'tanh':
            #print("Activate function: tanh")
            self.activate_function = Activation.tanh
            self.activate_function_derivative = Activation.tanh_derivative
        elif activate_function == 'relu':
            #print("Activate function: relu")
            self.activate_function = Activation.relu
            self.activate_function_derivative = Activation.relu_derivative
        else: 
            #print("Activate function: sigmoid")
            self.activate_function = Activation.sigmoid
            self.activate_function_derivative = Activation.sigmoid_derivative

        self.sigmoid = Activation.sigmoid
        self.sigmoid_derivative = Activation.sigmoid_derivative

        self.optimizer = optimizer
        # if self.optimizer == "adagrad":
        #     print("Optimizer: adagrad")
        # elif self.optimizer == "adam":
        #     print("Optimizer: adam")
        # else:
        #     print("Optimizer: SGD")
    
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.W = []
        self.b = []

        
    def forwardPass(self, X):
        """ Forward pass of MLP """
        # TODO
        self.result = [X]
        self.z_values = []
        
        A = X
        for i in range(len(self.W)):
            Z = np.dot(A, self.W[i]) + self.b[i]
            self.z_values.append(Z)
            if i == len(self.W) - 1:
                A = self.sigmoid(Z)
            else:
                A = self.activate_function(Z)
            self.result.append(A)
        return A

    def backwardPass(self, y):
        """ Backward pass of MLP """
        # TODO        
        y = y.reshape((len(y),1))
        m = y.shape[0]
        dW_list = []
        db_list = []
        e = self.result[-1] - y

        dZ = e * self.sigmoid_derivative(self.z_values[-1])

        for i in reversed(range(len(self.W))):
            dW = np.dot(self.result[i].T, dZ) * (1 / m)
            db = np.sum(dZ, axis=0, keepdims=True) * (1 / m)
            dW_list.append(dW)
            db_list.append(db)

            if i > 0:
                dZ = np.dot(dZ, self.W[i].T) * self.activate_function_derivative(self.z_values[i - 1])

        dW_list.reverse()
        db_list.reverse()

        return dW_list, db_list

    def update(self,dW_list,db_list):
        """ The update method to update parameters """
        # TODO
        Optimizer = optimizer()
        for i in range(len(self.W)):
            
            if self.optimizer == "adagrad":
                self.W[i], self.b[i], self.prev_W[i], self.prev_b[i] = Optimizer.adagrad(self.learning_rate, self.W[i], dW_list[i], self.b[i], db_list[i], self.prev_W[i], self.prev_b[i])
            elif self.optimizer == "adam":
                self.t += 1
                self.W[i], self.b[i], self.s_W[i], self.s_b[i], self.r_W[i], self.r_b[i] = Optimizer.adam(self.learning_rate, self.W[i], dW_list[i], self.b[i], db_list[i], 
                                                                                                          self.t, self.s_W[i], self.s_b[i], self.r_W[i], self.r_b[i])
            else:
                self.W[i], self.b[i] = Optimizer.SGD(self.learning_rate, self.W[i], dW_list[i], self.b[i], db_list[i])

    
    def fit(self, X_train, y_train):
        """ Fit method for MLP, call it to train your MLP model """
        # TODO
        new_layers = [X_train.shape[1]] + self.layers
        self.W = []
        self.b = []

        for i in range(1,len(new_layers)):
            self.W.append(np.random.randn(new_layers[i-1],new_layers[i]))
            self.b.append(np.zeros((1,new_layers[i])))
        
        self.prev_W = [np.zeros_like(w) for w in self.W]
        self.prev_b = [np.zeros_like(b) for b in self.b]

        self.s_W = [np.zeros_like(w) for w in self.W]
        self.s_b = [np.zeros_like(b) for b in self.b]
        self.r_W = [np.zeros_like(w) for w in self.W]
        self.r_b = [np.zeros_like(b) for b in self.b]
        self.t = 0

        for epoch in range(self.n_epoch):   
            self.forwardPass(X_train)
            dW_list, db_list = self.backwardPass(y_train)
            self.update(dW_list, db_list)
            
            if epoch % 100 == 0:
                loss = -np.mean(y_train * np.log(self.result[-1] + 1e-8) + (1 - y_train) * np.log(1 - self.result[-1] + 1e-8))
                # print(f"Epoch {epoch}/{self.n_epoch}, Loss: {loss}")

        
        
    def predict(self, X_test):
        """ Method for predicting class of the testing data """
        y_hat = self.predict_proba(X_test)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        return self.forwardPass(X_test)

# Decision Tree Classifier
class DecisionTreeClassifier(Classifier):
    def __init__(self, max_depth=6):
        self.max_depth = max_depth
        self.tree = None
        self.leaf_size_threshold = 5
        self.gain_threshold = 0.02

    def fit(self, X, y):
        self.tree = None
        #self.leaf_size_threshold = int(len(X) / 50)
        self.tree = self._grow_tree(X, y)
        #self._post_prune(self.tree, X, y)
        # self.print_tree()

    def _grow_tree(self, X, y, depth=0):
        
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.leaf_size_threshold:
            if np.sum(y) > len(y) - np.sum(y):
                leaf_value = 1
            else:
                leaf_value = 0
            return {"is_leaf": True, "value": leaf_value}
        
        best_split_feature, best_split_threshold = self.find_best_split(X, y)
        if best_split_feature is None or best_split_threshold is None:
            if np.sum(y) > len(y) - np.sum(y):
                leaf_value = 1
            else:
                leaf_value = 0
            return {"is_leaf": True, "value": leaf_value}
        

        X_left, X_right, y_left, y_right = self.split_dataset(X, y, best_split_feature, best_split_threshold)
        left_tree = self._grow_tree(X_left, y_left, depth + 1)
        right_tree = self._grow_tree(X_right, y_right, depth + 1)
        
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
    def find_best_split(self, X, y):
        best_split = [None, None] # feature_idx, threshold
        max_gain = self.gain_threshold

        parent_entropy = self.entropy(y)
        for feature_idx in range(X.shape[1]):            

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

# K-Nearest Neighbors Classifier
class KNearestNeighborClassifier(Classifier):
    def __init__(self, k=15, distance_metric='euclidean', p=3): 
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

# Naive Bayes Classifier
class NaiveBayesClassifier(Classifier):
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
        self.priors = {}
        self.likelihoods = {}
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

