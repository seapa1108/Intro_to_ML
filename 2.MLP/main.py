import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os


def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
    file = pd.read_csv('train_X.csv', index_col=0)
    preprocessor = Preprocessor(file)
    preprocessor.replace_string()

    preprocessor.replace_nan()
    preprocessor.outlier_handle()
    preprocessor.standarize()
    train_X = preprocessor.get_result()

    file = pd.read_csv('train_y.csv', index_col=0)
    preprocessor = Preprocessor(file)
    train_y = preprocessor.get_result()

    file = pd.read_csv('test_X.csv', index_col=0)
    preprocessor = Preprocessor(file)
    preprocessor.replace_string()
    
    preprocessor.replace_nan()
    preprocessor.outlier_handle()
    preprocessor.standarize()
    test_X = preprocessor.get_result()
    
    file = pd.read_csv('test_y.csv', index_col=0)
    preprocessor = Preprocessor(file)
    test_y = preprocessor.get_result()

    return train_X, train_y, test_X, test_y # train, test data should be numpy array


def main():
    
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index
    
    layers = [len(train_X[0]), 128, 8, 1]

    model = MLPClassifier(layers, "sigmoid", "SGD", learning_rate = 0.005, n_epoch = 5000) # remember to change the hyperparameter
    model.fit(train_X, train_y)
    pred = model.predict(test_X)

    acc = accuracy_score(pred, test_y)
    f1 = f1_score(pred, test_y, zero_division=0)
    mcc = matthews_corrcoef(pred, test_y)

    print(f'Acc: {acc:.5f}')
    print(f'F1 score: {f1:.5f}')
    print(f'MCC: {mcc:.5f}')
    scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'Scoring: {scoring:.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()