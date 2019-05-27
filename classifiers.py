import abc
from time import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import timeit


class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def context_interface(self, x, y, xt, yt):
        return self._strategy.algorithm_interface(x, y, xt, yt)

def Result(yTest,yPred):

    # normailze = true  If False, return the number of correctly classified samples.
    # Otherwise, return the fraction of correctly classified samples.
    acc = accuracy_score(yTest, yPred, normalize=True, sample_weight=None)
    # Build a text report showing the main classification metrics
    # (Ground truth (correct) target values, Estimated targets as returned by a classifier)
    cr = classification_report(yTest, yPred)
    print("Accuracy: ", acc)
    print("Classification Report: ", cr)

class Strategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def algorithm_interface(self, x, y, xt, yt):
        pass

class SvmAlgLinear(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):

        parameters = {'C': (0.1, 10, 100, 1e3, 1e4)}
        svc = SVC(kernel='linear')

        grid_search = GridSearchCV(svc, parameters, cv=3)
        grid_search.fit(xTrain, yTrain)

        print("Best Parameter: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        svc_best = grid_search.best_estimator_

        #accuracy = svc_best.score(X_test, Y_test)
        #print('The accuracy on testing set is: {0:.1f}%'.format(accuracy * 100))

class SvmAlgRbf(Strategy):
    def algorithm_interface(self, X_train, Y_train, X_test, Y_test):

        svc = SVC(kernel='rbf')
        parameters = {'C': (10, 100, 1e3, 1e4, 1e5), 'gamma': (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)}
        # gamma is a parameter
        # for non linear hyperplanes.The higher the gamma value it tries to exactly fit the training data set

        # C is the penalty parameter of the error term. It controls the trade off between smooth decision
        # boundary and classifying the training points correctly.

        grid_search = GridSearchCV(svc, parameters, cv=3)
        grid_search.fit(X_train, Y_train)

        print("Best Parameter: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        svc_best = grid_search.best_estimator_

        accuracy = svc_best.score(X_test, Y_test)
        print('The accuracy on testing set is: {0:.1f}%'.format(accuracy * 100))

        prediction = svc_best.predict(X_test)
        report = classification_report(Y_test, prediction)
        print(report)
        # Best Parameter:  {'C': 100000.0, 'gamma': 1e-07}
        # Best Score:  0.9423529411764706

class SvmAlgPoly(Strategy):
    def algorithm_interface(self, X_train, Y_train, X_test, Y_test):

        svc = SVC(kernel='poly')

        #degree is a parameter used when kernel is set to ‘poly’. It’s basically the degree of the polynomial used to find the hyperplane to split the data.
        parameters = { 'degree': (0,1,2,3,4,5,6)}

        grid_search = GridSearchCV(svc, parameters, cv=3)
        grid_search.fit(X_train, Y_train)

        print("Best Parameter: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        svc_best = grid_search.best_estimator_

        accuracy = svc_best.score(X_test, Y_test)
        print('The accuracy on testing set is: {0:.1f}%'.format(accuracy * 100))

        prediction = svc_best.predict(X_test)
        report = classification_report(Y_test, prediction)
        print(report)

class DecisionTree(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(xTrain, yTrain)
        yPred = clf.predict(xTest)
        Result(yTest,yPred)

class NaiveBayes(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):
        clf = GaussianNB()
        clf.fit(xTrain, yTrain)
        yPred = clf.predict(xTest)
        Result(yTest,yPred)

class KnnClassifier(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):

        xtotal = np.concatenate((x_train, x_test), axis=0)
        ytotal = np.concatenate((y_train, y_test), axis=0)

        k_range = range(1, 31)

        # list of scores from k_range
        k_scores = []

        # 1. we will loop through reasonable values of k
        for k in k_range:
            # 2. run KNeighborsClassifier with k neighbours
            knn = KNeighborsClassifier(n_neighbors=k)
            # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
            #cv = 10 fold cross validation
            scores = cross_val_score(knn, xtotal, ytotal, cv=10, scoring='accuracy')
            # 4. append mean of scores for k neighbors to k_scores list
            k_scores.append(scores.mean())
        print(k_scores)

        plt.plot(k_range, k_scores)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validated Accuracy')
        plt.show()

class RandomForest(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):

        #rfclassifier = RandomForestClassifier(n_estimators=100 ,random_state=0)
        rfclassifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state = 0)
        rfclassifier.fit(xTrain, yTrain)
        yPred = rfclassifier.predict(xTest)

        Result(yTest,yPred)

class LogRegression(Strategy):
    def algorithm_interface(self, xTrain, yTrain, xTest, yTest):
        log = LogisticRegression()
        log.fit(xTrain,yTrain)
        yPred=log.predict(xTest)
        print("Logestic Regression:")
        Result(yTest,yPred)
