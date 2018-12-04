import pandas as pd
import numpy as np
import seaborn as sns

from sklearnhelper import *
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



train_mix = pd.read_csv('train_mix.csv')
test_mix = pd.read_csv('test_mix.csv')

X_train = train_mix.loc[:,'Pclass':]
y_train = train_mix['Survived']
X_test = test_mix.loc[:,'Pclass':]

#######################################################################
# hyperparameter optimization
#######################################################################

#######################################################################
# n_estimators
get_n_estimators(RandomForestClassifier(), X_train, y_train, n=200, loops=20)
# 75
get_n_estimators(ExtraTreesClassifier(), X_train, y_train, n=200, loops=50)
# 100
get_n_estimators(AdaBoostClassifier(), X_train, y_train, n=200, loops=1)
# 75 (different seeds don't change accuracy)

# get_n_estimators doesn't play nice with GradientBoostingClassifier, so doing this instead
params = {'n_estimators': range(5,201,5)}
for clf in [GradientBoostingClassifier(), XGBClassifier()]
    score_df = pd.DataFrame()
    for i in range(10):
        clf.random_state = np.random.randint(999)
        grid_search = GridSearchCV(estimator = clf, param_grid=params, scoring='accuracy', n_jobs=-1, cv=5)
        grid_search.fit(X_train,y_train)
        scores = []
        scores = [x[1] for x in grid_search.grid_scores_]
        score_df[i] = scores
    plt.plot(score_df.mean(axis=1))
    plt.xlabel('num_trees')
    plt.ylabel('accuracy')
    plt.show()
# gbc 20, however gbc is robust to overfitting, so upping it to 50
# xgb 10
#######################################################################
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
#######################################################################
# tree parameters
rfc = RandomForestClassifier(n_estimators=75, random_state=0)
etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
abc = AdaBoostClassifier(n_estimators=75, random_state=0)
gbc = GradientBoostingClassifier(n_estimators=50, random_state=0)
# get n_estimators with early stopping
xgb = XGBClassifier()

# grid search doesn't always work very well when trying to optimize multiple
# parameters at once, so making a list of dicts for each classifier to pass
# to grid search iteratively
rfc_grid = [{'min_samples_leaf': np.linspace(1,10,num=10, dtype='int')}, {'max_depth': np.linspace(1,10,num=10, dtype='int')}, {'max_features':['auto', 'log2']}, {'criterion': ['gini', 'entropy']}]

etc_grid = [{'min_samples_leaf': np.linspace(1,10,num=10, dtype='int')}, {'max_depth': np.linspace(1,10,num=10, dtype='int')},{'max_features':['auto', 'log2']}, {'criterion': ['gini', 'entropy']}]

abc_grid =  {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10]}

gbc_grid = [{'min_samples_leaf': np.linspace(1,10,num=10, dtype='int')}, {'max_depth': np.linspace(1,10,num=10, dtype='int')}, {'max_features':['auto', 'log2']}, {'criterion': ['friedman_mse', 'mse', 'mae']}, {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10]}]

xgb_grid = [{'max_depth': np.linspace(1,10,num=10, dtype='int')}, {'subsample': np.linspace(.1,1,num=10)}, {'colsample_bytree': np.linspace(.1,1,num=10)}, {'reg_alpha': np.linspace(0,1,num=11)}, {'reg_lambda': np.linspace(0,1,num=11)}, {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10]}]

for clf, grid in zip([rfc, etc, abc, gbc, xgb], [rfc_grid, etc_grid, abc_grid, gbc_grid, xgb_grid]):
    for params in grid:
        grid_search_helper(clf, params)

# RandomForestClassifier
# {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 4}
# ExtraTreesClassifier
# {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 10}
# GradientBoostingClassifier
# {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 6}

# RandomForestClassifier
# gini:       train: 0.862    test: 0.829
# entropy:    train: 0.861    test: 0.832
# ExtraTreesClassifier
# gini:       train: 0.849    test: 0.831
# entropy:    train: 0.847    test: 0.828

# GradientBoostingClassifier
# friedman_mse:   train: 0.894    test: 0.829
# mse:            train: 0.895    test: 0.833
# mae:            train: 0.852    test: 0.829

# GradientBoostingClassifier
# {'learning_rate': 10}, train: 0.895, test: 0.833
# AdaBoostClassifier
# {'learning_rate': 10}, train: 0.849, test: 0.831

#######################################################################
# early stopping for xgboost
# https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/

x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=0)

# XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.1,
#        gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=7,
#        min_child_weight=1, missing=None, n_estimators=200, nthread=-1,
#        objective='binary:logistic', reg_alpha=0, reg_lambda=1,
#        scale_pos_weight=1, seed=0, silent=True, subsample=0.2)

eval_set = [(X_train, y_train), (X_val, y_val)]
xgb.fit(x_train, y_train, early_stopping_rounds=20, eval_set=eval_set, eval_metric="logloss", verbose=True)
results = xgb.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()
# [127]   validation_0-logloss:0.378978   validation_1-logloss:0.391348
# [128]   validation_0-logloss:0.379002   validation_1-logloss:0.39195
# [129]   validation_0-logloss:0.379218   validation_1-logloss:0.394324
# [130]   validation_0-logloss:0.37942    validation_1-logloss:0.395909
# Stopping. Best iteration:
# [110]   validation_0-logloss:0.383407   validation_1-logloss:0.3779
#######################################################################
# svc
svc = SVC(random_state=0)
svc_grid = {'C': np.logspace(-6, 4, num=11), 'gamma': np.logspace(-6, 4, num=11)}
grid_search_helper(svc, svc_grid)
# SVC
# {'C': 1000.0, 'gamma': 1.0e-05}, train: 0.836, test: 0.835

# also use early stopping to set max_iter if overfitting (which it's not here)
#######################################################################
# logistic regression
lr = LogisticRegression(random_state=0)
lr_grid = {'penalty': ['l1', 'l2'], 'C': np.logspace(-4, 4, num=9)}
grid_search_helper(lr, lr_grid)
# LogisticRegression
# {'C': 0.001, 'penalty': 'l1'}, train: 0.845, test: 0.828

#######################################################################
# KNN
knn = KNeighborsClassifier()
knn_grid = {'n_neighbors': np.linspace(1, 30, num=31, dtype='int'), 'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'], 'weights': ['uniform', 'distance']}
grid_search_helper(knn, knn_grid)
# KNeighborsClassifier
# {'algorithm': 'brute', 'n_neighbors': 18, 'weights': 'uniform'}, train: 0.859, test: 0.824
#######################################################################
# naive bayes
# principal component analysis

#######################################################################
# base predictions
#######################################################################
# params based on above optimization
rfc_params = {
    'n_jobs': -1,
    'n_estimators': 75,
    'max_depth': 15,
    'min_samples_leaf': 4,
    'criterion': 'entropy'
}

etc_params = {
    'n_jobs': -1,
    'n_estimators':100,
    'max_depth': 15,
    'min_samples_leaf': 10,
    'criterion': 'gini'
}

abc_params = {
    'n_estimators': 75,
    'learning_rate' : 10
}

gbc_params = {
    'n_estimators': 50,
    'max_depth': 5,
    'min_samples_leaf': 6,
    'learning_rate': 10,
    'criterion': 'mse'
}

xgb_params = {
    'n_estimators': 110,
    'colsample_bylevel': 1,
    'colsample_bytree': 0.2,
    'learning_rate': 0.001,
    'max_depth': 7,
    'subsample': 0.2
}

svc_params = {
    'C': 1000,
    'gamma': 0.00001
}

lr_params = {
    'C': 0.001,
    'penalty': 'l1'
}

knn_params = {
    'algorithm': 'brute',
    'n_neighbors': 18,
    'weights': 'uniform'
}

rfc = Trainer(clf=RandomForestClassifier, params=rfc_params)
etc = Trainer(clf=ExtraTreesClassifier, params=etc_params)
abc = Trainer(clf=AdaBoostClassifier, params=abc_params)
gbc = Trainer(clf=GradientBoostingClassifier, params=gbc_params)
svc = Trainer(clf=SVC, params=svc_params)
lr = Trainer(clf=LogisticRegression, params=lr_params)
knn = Trainer(clf=KNeighborsClassifier, params=knn_params)

X_train = X.values
y_train = y.values
X_test = X_test.values

rfc_oof_train, rfc_oof_test = rfc.get_oof(X_train, y_train, X_test) # Random Forest
etc_oof_train, etc_oof_test = etc.get_oof(X_train, y_train, X_test) # Extra Trees
abc_oof_train, abc_oof_test = abc.get_oof(X_train, y_train, X_test) # AdaBoost
gbc_oof_train, gbc_oof_test = gbc.get_oof(X_train, y_train, X_test) # Gradient Boost
svc_oof_train, svc_oof_test = svc.get_oof(X_train, y_train, X_test) # Support Vector Classifier
lr_oof_train, lr_oof_test = lr.get_oof(X_train, y_train, X_test) # Logistic Regressor
knn_oof_train, knn_oof_test = knn.get_oof(X_train, y_train, X_test) # K Nearest Neighbor

base_predictions_train = pd.DataFrame( {'RandomForest': rfc_oof_train.ravel(), 'ExtraTrees': etc_oof_train.ravel(), 'AdaBoost': abc_oof_train.ravel(),  'GradientBoost': gbc_oof_train.ravel(), 'SupportVector': svc_oof_train.ravel(), 'Logistic': lr_oof_train.ravel(), 'KNN': knn_oof_train.ravel()
    })
sns.heatmap(base_predictions_train.corr())

#######################################################################
# second level learning
#######################################################################

X_train = np.concatenate((rfc_oof_train, etc_oof_train, abc_oof_train, gbc_oof_train, svc_oof_train, lr_oof_train, knn_oof_train), axis=1)

X_test = np.concatenate((rfc_oof_test, etc_oof_test, abc_oof_test, gbc_oof_test, svc_oof_test, lr_oof_test, knn_oof_test), axis=1)

# xgb for second level learning
# params from example, optimize on base predictions
# outperforms optimized params above
gbm = XGBClassifier(
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(X_train, y_train)
