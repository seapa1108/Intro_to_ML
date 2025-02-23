import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import DecisionTreeClassifier
import os


def dataPreprocessing():
    # """ TODO, use your own dataPreprocess function here. """
     
    # return
    file = pd.read_csv('train_X.csv', index_col=0)
    preprocessor = Preprocessor(file)
    preprocessor.replace_string()
    preprocessor.replace_nan()
    train_X = preprocessor.get_result()

    file = pd.read_csv('train_y.csv', index_col=0)
    preprocessor = Preprocessor(file)
    train_y = preprocessor.get_result()    

    file = pd.read_csv('test_X.csv', index_col=0)
    preprocessor = Preprocessor(file)
    preprocessor.replace_string()
    preprocessor.replace_nan()
    test_X = preprocessor.get_result()

    return train_X, train_y, test_X # train, test data should be numpy array


def main():
    train_X, train_y, test_X = dataPreprocessing()

    # indices = np.arange(len(train_X))
    # np.random.shuffle(indices)
    
    # test_set_size = int(len(train_X) * ((86 / (426+86))))
    # test_indices = indices[:test_set_size]
    # train_indices = indices[test_set_size:]
    
    # train_X, test_X = train_X[train_indices], train_X[test_indices]
    # train_y, test_y = train_y[train_indices], train_y[test_indices]

    Decision_tree = DecisionTreeClassifier(max_depth=6)
    Decision_tree.fit(train_X,train_y)
    pred = Decision_tree.predict(test_X)
    Decision_tree.print_tree()

    df = pd.DataFrame({'label': pred})

    df.to_csv('pred.csv')
    
    # from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    # acc = accuracy_score(pred, test_y)
    # f1 = f1_score(pred, test_y, zero_division=0)
    # mcc = matthews_corrcoef(pred, test_y)

    # print(f'Acc: {acc:.5f}')
    # print(f'F1 score: {f1:.5f}')
    # print(f'MCC: {mcc:.5f}')
    # scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    # print(f'Scoring: {scoring:.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

