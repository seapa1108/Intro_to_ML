import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model.meta_learner import StackingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import os


def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
     
    file = pd.read_csv('train_x.csv', index_col=0)
    file2 = pd.read_csv('train_y.csv', index_col=0)

    preprocessor = Preprocessor(file)
    preprocessor.replace_string()
    preprocessor.replace_nan()
    drop = preprocessor.feature_selection(file2)
    preprocessor.drop_columns(drop)
    preprocessor.outlier_handle()
    preprocessor.standarize()
    train_X = preprocessor.get_result()

    preprocessor = Preprocessor(file2)
    train_y = preprocessor.get_result()
    

    file = pd.read_csv('test_x.csv', index_col=0)
    preprocessor = Preprocessor(file)
    preprocessor.replace_string()
    preprocessor.replace_nan()
    preprocessor.drop_columns(drop)
    preprocessor.outlier_handle()
    preprocessor.standarize()
    test_X = preprocessor.get_result()

    return train_X, train_y, test_X



def main():
    train_X, train_y, test_X = dataPreprocessing()
    
    num_folds = 10
    #fold = KFold(n_splits=10, random_state=42, shuffle=True)
    fold = KFold(n_splits=num_folds, random_state=42, shuffle=True)
    
    avg_acc, avg_f1, avg_mcc = 0, 0, 0

    model = StackingClassifier()
    for train_indices, test_indices in fold.split(train_X):
        fold_train_X, fold_test_X = train_X[train_indices], train_X[test_indices]
        fold_train_y, fold_test_y = train_y[train_indices], train_y[test_indices]

        
        model.fit(fold_train_X, fold_train_y)

        pred = model.predict(fold_test_X)

        avg_acc += accuracy_score(pred, fold_test_y)
        avg_f1 += f1_score(pred, fold_test_y, zero_division=0)
        avg_mcc += matthews_corrcoef(pred, fold_test_y)

        # a = accuracy_score(pred, fold_test_y)
        # b = f1_score(pred, fold_test_y, zero_division=0)
        # c = matthews_corrcoef(pred, fold_test_y)

        # print("\ncurrent fold average scoring:")
        # print(f'Acc: {a:.5f}')
        # print(f'F1 score: {b:.5f}')
        # print(f'MCC: {c:.5f}')

        # s = 0.3 * a + 0.35 * b + 0.35 * c
        # print(f'Scoring: {s:.5f}')

    avg_acc /= num_folds
    avg_f1 /= num_folds
    avg_mcc /= num_folds

    print("\nk-fold average scoring:")
    print(f'Acc: {avg_acc:.5f}')
    print(f'F1 score: {avg_f1:.5f}')
    print(f'MCC: {avg_mcc:.5f}')

    scoring = 0.3 * avg_acc + 0.35 * avg_f1 + 0.35 * avg_mcc
    print(f'Scoring: {scoring:.5f}')

    model.fit(train_X,train_y)
    pred = model.predict(test_X)

    df = pd.DataFrame({'label': pred})
    df.to_csv('pred.csv')

    # TODO 
    # build your Stacking model
    # predict the output of the testing data
    # remember to paste the result of K-fold CV to your report
    # remember to save the predict label as .csv file


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

