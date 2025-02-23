import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import KNearestNeighborClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import os
import matplotlib.pyplot as plt


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


def k_fold_cross_validation(train_X, train_y, num_folds=5, distance_metric='manhattan'):
    indices = np.arange(len(train_X))
    np.random.shuffle(indices)
    
    train_X = train_X[indices]
    train_y = train_y[indices]
    
    fold_size = len(train_X) // num_folds
    fold_indices = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(num_folds)]
    fold_indices[-1] = np.arange((num_folds - 1) * fold_size, len(train_X))

    error_rates = []

    k_values = range(1,50)

    for k in k_values:
        total_error = 0
        
        for i in range(num_folds):
            test_indices = fold_indices[i]
            train_indices = np.setdiff1d(np.arange(len(train_X)), test_indices)

            fold_train_X, fold_test_X = train_X[train_indices], train_X[test_indices]
            fold_train_y, fold_test_y = train_y[train_indices], train_y[test_indices]

            model = KNearestNeighborClassifier(k=k, distance_metric=distance_metric)
            model.fit(fold_train_X, fold_train_y)

            pred = model.predict(fold_test_X)

            total_error += np.mean(pred != fold_test_y.squeeze())

        avg_error = total_error / num_folds

        error_rates.append(avg_error)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, error_rates, marker='o', linestyle='-', color='red', label='Error Rate')
    plt.xlabel('k')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. k in kNN')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    train_X, train_y, test_X = dataPreprocessing()

    # k_fold_cross_validation(train_X, train_y, 20, distance_metric='manhattan')

    model = KNearestNeighborClassifier(k=48, distance_metric='manhattan')
    model.fit(train_X,train_y)

    pred = model.predict(test_X)
    #print(pred)

    df = pd.DataFrame({'label': pred})
    df.to_csv('pred.csv')
    
    

if __name__ == "__main__":
    np.random.seed(0)
    main()
    

