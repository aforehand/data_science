import pandas as pd
import numpy as np
import seaborn as sns

from sklearnhelper import *
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense


train_mix = pd.read_csv('train_mix.csv')
test_mix = pd.read_csv('test_mix.csv')

X_train = train_mix.loc[:,'Pclass':]
y_train = train_mix['Survived']
X_test = test_mix.loc[:,'Pclass':]
IDs = test_mix.PassengerId

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)


lr_params = {}
linear_svc_params = {'kernel': 'linear'}
rbf_svc_params = {}
knn_params = {}
# nn_1_params = {}
# nn_2_params = {}
# et_params = {}
rf_params = {}
ada_params = {'n_estimators': 100}
xgb_params = {}
gbc_params = {}

# TODO: include nn and et params if/when it makes sense
params = [lr_params, linear_svc_params, rbf_svc_params, knn_params, rf_params, ada_params, xgb_params, gbc_params]

models = [LogisticRegression, SVC, SVC, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, XGBClassifier, GradientBoostingClassifier]

labels = ['logistic', 'linear_svc', 'nonlinear_svc', 'random_forest', 'ada_boost', 'xgb', 'gradient_boosting']


# TODO: create diverse datasets. see ensembling.text
#######################################################################
# TODO: rework grid_search_helper and get_n_estimators to find the point where the test set stops improving vs. where the diffrence between train and test is smallest

# grid search helper
def grid_search_helper(clf, params, X=X_train, y=y_train):
    print(clf)
    grid_search = GridSearchCV(estimator=clf, param_grid=params, scoring='accuracy', n_jobs=-1, cv=5)
    grid_search.fit(X,y)
    grid_search_df = pd.DataFrame(data={'params': grid_search.cv_results_['params'], 'train': grid_search.cv_results_['mean_train_score'], 'test': grid_search.cv_results_['mean_test_score']})
    grid_search_df.sort_values('test', ascending=False, inplace=True)
    grid_search_df['fit'] = grid_search_df.train - grid_search_df.test
    best_params=grid_search_df.params[grid_search_df.fit==grid_search_df.fit.min()]
    for i in range(best_params.size):
        print("{}, train: {:.3f}, test: {:.3f}".format(best_params.iloc[i], grid_search_df.train.iloc[i], grid_search_df.test.iloc[i]))
    return best_params.iloc[0]

#######################################################################
# get n_estimators
def get_n_estimators(clf, estimators=np.linspace(10, 500, num=50, dtype=int)):
    train_scores = []
    test_scores = []
    for n in estimators:
        clf.n_estimators = n
        clf.fit(X_tr, y_tr)
        train_scores.append(clf.score(X_tr, y_tr))
        test_scores.append(clf.score(X_val, y_val))

    score_df = pd.DataFrame(data={'estimators': estimators, 'train': train_scores, 'test': test_scores})
    score_df['diff'] = score_df.train - score_df.test
    print(score_df.sort_values(by='diff').iloc[0])
    print(score_df.sort_values(by='test', ascending=False).iloc[0])

#######################################################################
# linear models
#######################################################################
# LogisticRegression / RidgeRegression
lr = LogisticRegression(random_state=0)

for C in np.logspace(-5,1,num=7):
    lr.C = C
    lr.fit(X_tr, y_tr)
    print('C: {}; train: {:.3f}; test: {:.3f}'.format(C, lr.score(X_tr, y_tr), lr.score(X_val, y_val)))
# C: 1.0; train: 0.840; test: 0.825
lr_params.update(){'C': 1.0})

# Linear SVM
linear_svc = SVC(kernel='linear', random_state=0)
for C in np.logspace(-5,1,num=7):
    linear_svc.C = C
    linear_svc.fit(X_tr, y_tr)
    print('C: {}; train: {:.3f}; test: {:.3f}'.format(C, linear_svc.score(X_tr, y_tr), linear_svc.score(X_val, y_val)))
# C: 1.0; train: 0.835; test: 0.825
linear_svc_params.update({'C': 1.0})

#######################################################################
# nonlinear SVM
#######################################################################
rbf_svc = SVC(random_state=0)
svc_grid = {'C': np.logspace(-6, 4, num=11), 'gamma': np.logspace(-6, 4, num=11)}
best_params = grid_search_helper(rbf_svc, svc_grid)
# {'C': 1000.0, 'gamma': 1.0e-05}, train: 0.836, test: 0.835
# rbf_svc_params.update(best_params)
rbf_svc_params.update({'C': 1000.0, 'gamma': 1.0e-05})

#######################################################################
# KNN
#######################################################################
knn = KNeighborsClassifier()
knn_grid = {'n_neighbors': np.linspace(1, 30, num=31, dtype='int'), 'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'], 'weights': ['uniform', 'distance']}
best_params = grid_search_helper(knn, knn_grid)
# {'algorithm': 'brute', 'n_neighbors': 18, 'weights': 'uniform'}, train: 0.859, test: 0.824
# knn_params.update(best_params)
knn_params.update({'algorithm': 'brute', 'n_neighbors': 18, 'weights': 'uniform'})
#######################################################################
# Neural Nets
# different depths
#######################################################################
# Keras
nn_1 = Sequential()
nn_1.add(Dense(64, activation='relu', input_dim=21))
nn_1.add(Dense(1, activation='sigmoid'))
nn_1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# nn_1.fit(X_tr, y_tr, epochs=20, batch_size=128)

nn_2 = Sequential()
nn_2.add(Dense(64, activation='relu', input_dim=21))
nn_2.add(Dense(64, activation='relu', input_dim=21))
nn_2.add(Dense(64, activation='relu', input_dim=21))
nn_2.add(Dense(1, activation='sigmoid'))
nn_2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# nn_2.fit(X_tr, y_tr, epochs=20, batch_size=128)

# pytorch

#######################################################################
# Factorization Machine
# useful for large, sparse datasets
#######################################################################
# libfm

#######################################################################
# bagging
#######################################################################
 # ExtraTrees
 # TODO: implement

 # RandomForest
rf = RandomForestClassifier()
get_n_estimators(rf)
# test            0.829596
# train           0.926647
# diff            0.097050
# estimators    100.000000
# Name: 9, dtype: float64
# test            0.829596
# train           0.926647
# diff            0.097050
# estimators    100.000000
# Name: 9, dtype: float64

for n in np.linspace(1,10,num=10,dtype=int):
    rf.min_samples_leaf = n
    rf.fit(X_tr, y_tr)
    print('min_samples_leaf: {}; train: {:.3f}; test: {:.3f}'.format(n, rf.score(X_tr, y_tr), rf.score(X_val, y_val)))
# min_samples_leaf: 2; train: 0.886; test: 0.843
rf_params.update({'n_estimators': 100, 'min_samples_leaf': 2})

#######################################################################
# boosting
# different implementations, different depths
#######################################################################
# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=0)

for rate in np.logspace(-5,1,num=7):
    ada.learning_rate = rate
    ada.fit(X_tr, y_tr)
    print('learning_rate: {}; train: {:.3f}; test: {:.3f}'.format(rate, ada.score(X_tr, y_tr), ada.score(X_val, y_val)))
# learning_rate: 1.0; train: 0.840; test: 0.843

for factor in [1,2,3,4,5,10]:
    ada.n_estimators = 100*factor
    ada.learning_rate = 1/factor
    ada.fit(X_tr, y_tr)
    print('factor: {}; train: {:.3f}; test: {:.3f}'.format(factor, ada.score(X_tr, y_tr), ada.score(X_val, y_val)))
# factor: 2; train: 0.846; test: 0.843
ada_params.update({'n_estimators': 200, 'learning_rate':  0.5})

# XGBoost
xgb = XGBClassifier()
for rate in np.logspace(-5,1,num=7):
    xgb.learning_rate = rate
    xgb.fit(X_tr, y_tr)
    print('learning_rate: {}; train: {:.3f}; test: {:.3f}'.format(rate, xgb.score(X_tr, y_tr), xgb.score(X_val, y_val)))
# learning_rate: 0.1; train: 0.870; test: 0.830
xgb_params.update({'learning_rate': 0.1})

for factor in [1,2,3,4,5,10]:
    xgb.n_estimators = 100*factor
    xgb.learning_rate = .1/factor
    xgb.fit(X_tr, y_tr)
    print('factor: {}; train: {:.3f}; test: {:.3f}'.format(factor, xgb.score(X_tr, y_tr), xgb.score(X_val, y_val)))
# factor: 1; train: 0.870; test: 0.830

for w in np.linspace(1,50,num=10,dtype=int):
    xgb.min_child_weight = w
    xgb.fit(X_tr, y_tr)
    print('weight: {}; train: {:.3f}; test: {:.3f}'.format(w, xgb.score(X_tr, y_tr), xgb.score(X_val, y_val)))
# weight: 17; train: 0.853; test: 0.834
for w in np.linspace(7,15,num=9,dtype=int):
    xgb.min_child_weight = w
    xgb.fit(X_tr, y_tr)
    print('weight: {}; train: {:.3f}; test: {:.3f}'.format(w, xgb.score(X_tr, y_tr), xgb.score(X_val, y_val)))
# weight: 15; train: 0.838; test: 0.839
xgb_params.update({'min_child_weight': 15, 'max_depth': 7})

# GradientBoost
gbc = GradientBoostingClassifier()
# defaults for n_estimators and learning_rate are fine
for d in np.linspace(2,20,num=10,dtype=int):
    gbc.max_depth = d
    gbc.fit(X_tr, y_tr)
    print('depth: {}; train: {:.3f}; test: {:.3f}'.format(d, gbc.score(X_tr, y_tr), gbc.score(X_val, y_val)))
# depth: 4; train: 0.907; test: 0.848
gbc_params.update({'max_depth': 4})

for w in np.linspace(0,0.5,num=11):
    gbc.min_weight_fraction_leaf = w
    gbc.fit(X_tr, y_tr)
    print('weight: {}; train: {:.3f}; test: {:.3f}'.format(w, gbc.score(X_tr, y_tr), gbc.score(X_val, y_val)))
# weight: 0.05; train: 0.870; test: 0.848
# gbc.min_weight_fraction_leaf=0.05
gbc_params.update({'max_depth': 4, 'min_weight_fraction_leaf': 0.05})

# hightgbm, h20's gbm, catboost, dart

#######################################################################
# base predictions
#######################################################################
# TODO: outputs full train set for train_predictions, so probably overfitting meta_model. figure out if can kfold meta model or must rework get_oof
# TODO: predict_proba in get_oof?
train_predictions = []
test_predictions = []

for m, p in zip(models, params):
    train, test = Trainer(clf=m,params=p).get_oof(X_train.values, y_train.values, X_test.values)

    train_predictions.append(train.ravel())
    test_predictions.append(test.ravel())


#######################################################################
# meta predictions
#######################################################################

meta_train = pd.DataFrame(dict(zip(labels, train_predictions)))
meta_test = pd.DataFrame(dict(zip(labels, test_predictions)))

sns.heatmap(meta_train.corr())

# TODO: generate meta features, pairwise diffrences, row-wise stats

# TODO: tune second level model params
# TODO: make the input_dim not a magic number
# TODO: figure out how to get binary output from nn
# meta_nn = Sequential()
# meta_nn.add(Dense(10, activation='relu', input_dim=7))
# meta_nn.add(Dense(1, activation='sigmoid'))
# meta_nn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# meta_nn.fit(meta_train, y_train, epochs=20)

# TODO: train with kfold?
meta_model = GradientBoostingClassifier()
meta_model.fit(meta_train, y_train)

predictions = meta_nn.predict(meta_test)
results = pd.concat([IDs, pd.Series(predictions, name='Survived')], axis=1)
results.to_csv('titanic5.csv', index=False)




# TODO: make param tuning more automatic somehow
