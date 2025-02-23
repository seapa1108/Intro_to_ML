import numpy as np
import pandas as pd

from preprocessor import Preprocessor
from model import LogisticRegressionClassifier
from sklearn.metrics import accuracy_score

def dataPreprocessing():
    """ TODO, implement your own dataPreprocess function here. """

    file = pd.read_csv('train_X.csv', index_col=0)
    preprocessor = Preprocessor(file)
    preprocessor.replace_string()

    # choose a way to impute replace_nan or mice_imputation
    preprocessor.replace_nan()
    # preprocessor.mice_imputation()

    preprocessor.outlier_handle()
    preprocessor.standarize()
    train_X = preprocessor.get_result()

    file = pd.read_csv('train_y.csv', index_col=0)
    preprocessor = Preprocessor(file)
    train_y = preprocessor.get_result()

    file = pd.read_csv('test_X.csv', index_col=0)
    preprocessor = Preprocessor(file)
    preprocessor.replace_string()
    
    # choose a way to impute replace_nan or mice_imputation
    preprocessor.replace_nan()
    # preprocessor.mice_imputation()

    preprocessor.outlier_handle()
    preprocessor.standarize()
    test_X = preprocessor.get_result()
    
    file = pd.read_csv('test_y.csv', index_col=0)
    preprocessor = Preprocessor(file)
    test_y = preprocessor.get_result()

    return train_X, train_y, test_X, test_y # train, test data should be numpy array


def main():
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index
    model = LogisticRegressionClassifier()
    model.fit(train_X, train_y)

    pred = model.predict(test_X)
    prob = model.predict_proba(test_X)
    prob = [f'{x:.5f}' for x in prob]
    # print(f'Prob: {prob}')
    print(f'Acc: {accuracy_score(pred, test_y):.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

