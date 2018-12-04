import pandas as pd
import numpy as np

train_df = pd.read_csv('titanic/train.csv')
y = train_df.Survived
X = train_df.loc[:,'Pclass':]
ids = train_df.PassengerId

numeric = ['Age', 'Fare']
count = ['SibSp', 'Parch']
categorical = ['Sex', 'Cabin', 'Embarked']
ordinal = ['Pclass']
text = ['Name']
generated = []
boolean = []


X.drop('Ticket', axis=1, inplace=True)

#######################################################################
# preprocessing
#######################################################################

# discretize the numeric data
# quarts = [0.420, 20.125, 28.000, 38.000, 80.000]
X['Age'] = pd.qcut(X.Age, 4, labels=False)

# reduce cabin to deck
X.Cabin.replace('[0-9]+.*', '', regex=True, inplace=True)

# retain only honorifics in names, grouping uncommon titles
X.Name.replace('^(.+), ', '', regex=True, inplace=True)
X.Name.replace('\..*', '.', regex=True, inplace=True)
names = {'Mr.': 'Mr', 'Mrs.': 'Mrs', 'Miss.': 'Miss', 'Master.': 'Master', 'Don.': 'Uncommon', 'Rev.': 'Uncommon', 'Dr.': 'Uncommon', 'Mme.': 'Uncommon','Ms.': 'Uncommon', 'Major.': 'Uncommon', 'Lady.': 'Uncommon', 'Sir.': 'Uncommon', 'Mlle.': 'Uncommon', 'Col.': 'Uncommon', 'Capt.': 'Uncommon', 'the Countess.': 'Uncommon', 'Jonkheer.': 'Uncommon'}
X['Name'] = X.Name.map(names)

#######################################################################
# feature generation
#######################################################################

X['FamSize'] = X.SibSp+X.Parch
count.append('FamSize')

X['IsAlone'] = (X.FamSize == 0.0)
boolean.append('IsAlone')

X['HasDependents'] = (X.Age>0) & (X.Age<3) & (X.Parch>0)
boolean.append('HasDependents')

X['SexClass'] = X.Sex+X.Pclass.astype(str)
generated.append('SexClass')

X['AgeClass'] = X.Age.astype(str)+X.Pclass.astype(str)
generated.append('AgeClass')

X['FamClass'] = X.FamSize.astype(str)+X.Pclass.astype(str)
generated.append('FamClass')

X['AgeDeck'] = X.Age.astype(str)+X.Cabin
X.AgeDeck[X.Age.isnull()] = np.nan
generated.append('AgeDeck')

X['SexDeck'] = X.Sex+X.Cabin
generated.append('SexDeck')

X['SexFam'] = X.Sex+X.IsAlone.astype(str)
generated.append('SexFam')

X['ClassDeck'] = X.Pclass.astype(str)+X.Cabin
generated.append('ClassDeck')
#######################################################################
# encoding
#######################################################################
from sklearn.preprocessing import LabelEncoder

#making a copy of the DataFrame
x_en = X.copy()
le = LabelEncoder()
for col in categorical+text+generated+boolean:
    x_en.loc[:,col] = le.fit_transform(X.loc[:,col].astype(str))
    # fillna categorical
    if col in X.isnull().any().index:
        null_code = x_en[col].max()
        x_en[col].replace(null_code, -999, inplace=True)

#fillna numeric
x_en.fillna(-999, inplace=True)

#######################################################################
# validation
#######################################################################
from sklearn.ensemble import RandomForestClassifier

x_train, x_val, y_train, y_val = train_test_split(x_en, y, random_state=0)
forest = RandomForestClassifier(random_state=0).fit(x_train, y_train)
forest.score(x_train, y_train)
# 0.948
forest.score(x_val, y_val)
# 0.803

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0).fit(x_train, y_train)
gbc.score(x_train, y_train)
# 0.895
gbc.score(x_val, y_val)
# 0.812
#######################################################################
# more features, mean encodings
# same through line 48
#######################################################################

X['HasCabin'] = ~X.Cabin.isnull()
boolean.append('HasCabin')

X['AgeSex'] = X.Age.astype(str)+X.Sex.astype(str)
generated.append('AgeSex')

X['ClassCabin'] = X.Pclass.astype(str)+X.HasCabin.astype(str)
generated.append('ClassCabin')

X['SexCabin'] = X.Sex.astype(str)+X.HasCabin.astype(str)
generated.append('SexCabin')

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=0)

# data with targets for calculating means
all_data = x_train.copy()
all_data['target'] = y_train.copy()


x_en = X.copy()

# quick and dirty mean encode
for col in categorical+text+generated+boolean:
    means = all_data.groupby(col).target.mean()
    x_en[col] = x_en[col].map(means)

# fillnas
x_en.Age.fillna(-999, inplace=True)
x_en.Cabin.fillna(-999, inplace=True)
x_en.Embarked.fillna(-999, inplace=True)
x_en.FamClass.fillna(y_train.mean(), inplace=True)
x_en.AgeDeck.fillna(y_train.mean(), inplace=True)
x_en.SexDeck.fillna(y_train.mean(), inplace=True)
x_en.ClassDeck.fillna(y_train.mean(), inplace=True)

# validate
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
clf = XGBClassifier()
scores = cross_val_score(clf, x_en, y)
scores.mean()
# 0.824
# (label encoded: 0.814)
#######################################################################
# with ensembling
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
#######################################################################
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import seaborn as sns
from sklearnhelper import Trainer


X_en['Fare'] = pd.qcut(X.Fare, 4, labels=False)

sns.heatmap(x_en.astype(float).corr())
# determined that a lot of the generated features are correlated, so removed them:
X_en.drop(['SexClass', 'SexDeck', 'FamClass', 'SexFam', 'SexCabin', 'AgeDeck', 'ClassDeck', 'ClassCabin'], axis=1, inplace=True)

x_train, x_val, y_train, y_val = train_test_split(X_en, y, random_state=0)
# these require arrays
x_train = x_train.values
x_val = x_val.values
y_train = y_train.values
y_val = y_val.values

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

rf = Trainer(clf=RandomForestClassifier, params=rf_params)
et = Trainer(clf=ExtraTreesClassifier, params=et_params)
ada = Trainer(clf=AdaBoostClassifier, params=ada_params)
gb = Trainer(clf=GradientBoostingClassifier, params=gb_params)
svc = Trainer(clf=SVC, params=svc_params)

et_oof_train, et_oof_val = et.get_oof(x_train, y_train, x_val) # Extra Trees
rf_oof_train, rf_oof_val = rf.get_oof(x_train, y_train, x_val) # Random Forest
ada_oof_train, ada_oof_val = ada.get_oof(x_train, y_train, x_val) # AdaBoost
gb_oof_train, gb_oof_val = gb.get_oof(x_train, y_train, x_val) # Gradient Boost
svc_oof_train, svc_oof_val = svc.get_oof(x_train, y_train, x_val) # Support Vector Classifier

rf_features = rf.feature_importances(x_train,y_train)
et_features = et.feature_importances(x_train, y_train)
ada_features = ada.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train,y_train)

# for some reason gotta assign the lists manually
# feats = <output>.split(' ')
# feats = list(filter(None, feats))
# feats = ','.join(feats)

# for plotting
importance_df = pd.DataFrame({'features': X_en.columns.values, 'random_forest': rf_features, 'extra_trees': et_features, 'adaboost': ada_features, 'gradient_boost': gb_features})
importance_df['mean'] = importance_df.mean(axis=1)
feature_importances(importance_df['mean'].values, X_en.columns)

# put base predictions in a DataFrame
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(), 'ExtraTrees': et_oof_train.ravel(), 'AdaBoost': ada_oof_train.ravel(),  'GradientBoost': gb_oof_train.ravel(), 'SupportVector': svc_oof_train.ravel()
    })

sns.heatmap(base_predictions_train.corr())

# create new train/val arrays for second level learning
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_val = np.concatenate(( et_oof_val, rf_oof_val, ada_oof_val, gb_oof_val, svc_oof_val), axis=1)

# train an XGBClassifier on the second level data
gbm = XGBClassifier(
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_val)

gbm.score(x_val, y_val)
# 0.821

#######################################################################
# next steps
#######################################################################

# implement a good cross-validation strategy in training the models to find
# optimal parameter values
# introduce a greater variety of base models. the more uncorrelated the base
# models, the better the results

# possible feature changes
# more age groups
# look at each feature more carefully to see if there are more appropriate
# values to for fillna
# there are many ages missing, look at average/most common age for titles to
# guess missing values
# create new features instead of replacing original ones for wacky encodings
# compare performance for mean- vs label-encoded
# look into which models require regularization, maybe good to do?
