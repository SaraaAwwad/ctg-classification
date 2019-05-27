import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import classifiers as cs


def load ():

    df = pd.read_excel('CTG.xls', "Sheet1")

    # start from 3 and skip last two cols (class and nsp)
    X = df.ix[1:2126, 3:-2].values

    # last col (class)
    Y = df.ix[1:2126, -1].values
    return X, Y

if __name__ == '__main__':
    print("Loading data...")
    X,Y = load()
    print("Data loaded.")


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    concrete_strategy_a = cs.RandomForest()
    context = cs.Context(concrete_strategy_a)
    context.context_interface(x_train, y_train, x_test, y_test)
