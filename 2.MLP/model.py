import numpy as np
from abc import ABC, abstractmethod

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


class MLPClassifier(Classifier):
    def __init__(self, layers, activate_function, optimizer, learning_rate , n_epoch = 1000):
        """ TODO, Initialize your own MLP class """

        self.layers = layers

        Activation = activation()
        if activate_function == 'tanh':
            print("Activate function: tanh")
            self.activate_function = Activation.tanh
            self.activate_function_derivative = Activation.tanh_derivative
        elif activate_function == 'relu':
            print("Activate function: relu")
            self.activate_function = Activation.relu
            self.activate_function_derivative = Activation.relu_derivative
        else: 
            print("Activate function: sigmoid")
            self.activate_function = Activation.sigmoid
            self.activate_function_derivative = Activation.sigmoid_derivative

        self.sigmoid = Activation.sigmoid
        self.sigmoid_derivative = Activation.sigmoid_derivative

        self.optimizer = optimizer
        if self.optimizer == "adagrad":
            print("Optimizer: adagrad")
        elif self.optimizer == "adam":
            print("Optimizer: adam")
        else:
            print("Optimizer: SGD")
    
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.W = []
        self.b = []

        for i in range(1,len(layers)):
            self.W.append(np.random.randn(layers[i-1],layers[i]))
            self.b.append(np.zeros((1,layers[i])))
        
        self.prev_W = [np.zeros_like(w) for w in self.W]
        self.prev_b = [np.zeros_like(b) for b in self.b]

        self.s_W = [np.zeros_like(w) for w in self.W]
        self.s_b = [np.zeros_like(b) for b in self.b]
        self.r_W = [np.zeros_like(w) for w in self.W]
        self.r_b = [np.zeros_like(b) for b in self.b]
        self.t = 0

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
        m = y.shape[1]
        dW_list = []
        db_list = []
        e = self.result[-1] - y
        dZ = e * self.sigmoid_derivative(self.z_values[-1])

        for i in reversed(range(len(self.W))):
            dW = np.dot(self.result[i].T, dZ) * (1 / m)
            db = np.sum(dZ, axis=0, keepdims=True) * (1 / m)
            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if i > 0:
                dZ = np.dot(dZ, self.W[i].T) * self.activate_function_derivative(self.z_values[i - 1])

        return dW_list, db_list

    def update(self,dW_list,db_list):
        """ The update method to update parameters """
        # TODO
        for i in range(len(self.W)):
            Optimizer = optimizer()
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


    