import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

SEED = 0
NFOLDS = 5

#TODO: update to work with regressors as well as classifiers
class Trainer(object):
    def __init__(self, clf, seed=SEED, params=None):
        # try:
        #     params['random_state'] = seed
        # except:
        #     pass
        self.clf = clf(**params)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        return self.clf.predict(X)

    def fit(self,X,y):
        return self.clf.fit(X,y)

    def feature_importances(self,X,y):
        return self.clf.fit(X,y).feature_importances_

    # out of fold predictions
    def get_oof(self, X_train, y_train, X_test):
        ntrain = X_train.shape[0]
        ntest = X_test.shape[0]
        kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            X_tr = X_train[train_index]
            y_tr = y_train[train_index]
            X_val = X_train[test_index]

            self.train(X_tr, y_tr)

            oof_train[test_index] = self.predict(X_val)
            oof_test_skf[i, :] = self.predict(X_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
