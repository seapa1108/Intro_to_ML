import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import os

def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
    file = pd.read_csv('train_X.csv', index_col=0)
    file2 = pd.read_csv('train_y.csv', index_col=0)

    preprocessor = Preprocessor(file)
    preprocessor.replace_string()
    preprocessor.replace_nan()
    drop = preprocessor.feature_selection(file2)
    preprocessor.drop_columns(drop)
    preprocessor.standarize()
    train_X = preprocessor.get_result()

    preprocessor = Preprocessor(file2)
    train_y = preprocessor.get_result()
    

    file = pd.read_csv('test_X.csv', index_col=0)
    preprocessor = Preprocessor(file)
    preprocessor.replace_string()
    preprocessor.replace_nan()
    preprocessor.drop_columns(drop)
    preprocessor.standarize()
    test_X = preprocessor.get_result()

    return train_X, train_y, test_X

def k_fold_cross_validation(train_X, train_y, num_folds):
    indices = np.arange(len(train_X))
    np.random.shuffle(indices)
    
    train_X = train_X[indices]
    train_y = train_y[indices]
    
    fold_size = len(train_X) // num_folds
    fold_indices = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(num_folds)]
    fold_indices[-1] = np.arange((num_folds - 1) * fold_size, len(train_X))

    avg_acc = 0
    avg_f1 = 0
    avg_mcc = 0
        
    for i in range(num_folds):
        test_indices = fold_indices[i]
        train_indices = np.setdiff1d(np.arange(len(train_X)), test_indices)

        fold_train_X, fold_test_X = train_X[train_indices], train_X[test_indices]
        fold_train_y, fold_test_y = train_y[train_indices], train_y[test_indices]

        model = NaiveBayesClassifier()
        model.fit(fold_train_X, fold_train_y)

        pred = model.predict(fold_test_X)

        avg_acc += accuracy_score(pred, fold_test_y)
        avg_f1 += f1_score(pred, fold_test_y, zero_division=0)
        avg_mcc += matthews_corrcoef(pred, fold_test_y)

    avg_acc = avg_acc / num_folds
    avg_f1 = avg_f1 / num_folds
    avg_mcc = avg_mcc / num_folds
    
    print("k-fold average scoring:")
    print(f'Acc: {avg_acc:.5f}')
    print(f'F1 score: {avg_f1:.5f}')
    print(f'MCC: {avg_mcc:.5f}')
    scoring = 0.3 * avg_acc + 0.35 * avg_f1 + 0.35 * avg_mcc
    print(f'Scoring: {scoring:.5f}')
    

def main():
    train_X, train_y, test_X = dataPreprocessing()

    k_fold_cross_validation(train_X, train_y, 10)

    model = NaiveBayesClassifier()
    model.fit(train_X,train_y)

    pred = model.predict(test_X)

    df = pd.DataFrame({'label': pred})
    df.to_csv('pred.csv')


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

