import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearnhelper import SklearnHelper

train_df = pd.read_csv('train.csv')
y_train = train_df.Survived
X_train = train_df.loc[:,'Pclass':]

numeric = ['Age', 'Fare']
count = ['SibSp', 'Parch']
categorical = ['Sex', 'Cabin', 'Embarked']
ordinal = ['Pclass']
text = ['Name']
generated = []
boolean = []

#######################################################################
# preprocessing
#######################################################################

X_train.drop('Ticket', axis=1, inplace=True)

# discretize the numeric data
# quarts = [0.420, 20.125, 28.000, 38.000, 80.000]
X_train['Age'] = pd.qcut(X_train.Age, 4, labels=False)
X_train['Fare'] = pd.qcut(X_train.Fare, 4, labels=False)

# reduce cabin to deck
X_train.Cabin.replace('[0-9]+.*', '', regex=True, inplace=True)

# retain only honorifics in names, grouping uncommon titles
X_train.Name.replace('^(.+), ', '', regex=True, inplace=True)
X_train.Name.replace('\..*', '.', regex=True, inplace=True)
names = {'Mr.': 'Mr', 'Mrs.': 'Mrs', 'Miss.': 'Miss', 'Master.': 'Master', 'Don.': 'Uncommon', 'Rev.': 'Uncommon', 'Dr.': 'Uncommon', 'Mme.': 'Uncommon','Ms.': 'Uncommon', 'Major.': 'Uncommon', 'Lady.': 'Uncommon', 'Sir.': 'Uncommon', 'Mlle.': 'Uncommon', 'Col.': 'Uncommon', 'Capt.': 'Uncommon', 'the Countess.': 'Uncommon', 'Jonkheer.': 'Uncommon'}
X_train['Name'] = X_train.Name.map(names)

#######################################################################
# feature generation
#######################################################################

X_train['FamSize'] = X_train.SibSp+X_train.Parch
count.append('FamSize')
X_train['IsAlone'] = (X_train.FamSize == 0.0)
boolean.append('IsAlone')
X_train['HasDependents'] = (X_train.Age>0) & (X_train.Age<3) & (X_train.Parch>0)
boolean.append('HasDependents')
X_train['AgeClass'] = X_train.Age.astype(str)+X_train.Pclass.astype(str)
generated.append('AgeClass')
X_train['HasCabin'] = ~X_train.Cabin.isnull()
boolean.append('HasCabin')
X_train['AgeSex'] = X_train.Age.astype(str)+X_train.Sex.astype(str)
generated.append('AgeSex')

#######################################################################
# test data
#######################################################################

test_df = pd.read_csv('test.csv')
X_test = test_df.loc[:,'Pclass':]
ids = test_df.PassengerId

X_test.drop('Ticket', axis=1, inplace=True)
X_test['Age'] = pd.qcut(X_test.Age, 4, labels=False)
X_test['Fare'] = pd.qcut(X_test.Fare, 4, labels=False)
X_test.Cabin.replace('[0-9]+.*', '', regex=True, inplace=True)
X_test.Name.replace('^(.+), ', '', regex=True, inplace=True)
X_test.Name.replace('\..*', '.', regex=True, inplace=True)
names = {'Mr.': 'Mr', 'Mrs.': 'Mrs', 'Miss.': 'Miss', 'Master.': 'Master', 'Don.': 'Uncommon', 'Rev.': 'Uncommon', 'Dr.': 'Uncommon', 'Mme.': 'Uncommon','Ms.': 'Uncommon', 'Major.': 'Uncommon', 'Lady.': 'Uncommon', 'Sir.': 'Uncommon', 'Mlle.': 'Uncommon', 'Col.': 'Uncommon', 'Capt.': 'Uncommon', 'the Countess.': 'Uncommon', 'Jonkheer.': 'Uncommon'}
X_test['Name'] = X_test.Name.map(names)


X_test['FamSize'] = X_test.SibSp+X_test.Parch
X_test['IsAlone'] = (X_test.FamSize == 0.0)
X_test['HasDependents'] = (X_test.Age>0) & (X_test.Age<3) & (X_test.Parch>0)
X_test['AgeClass'] = X_test.Age.astype(str)+X_test.Pclass.astype(str)
X_test['HasCabin'] = ~X_test.Cabin.isnull()
X_test['AgeSex'] = X_test.Age.astype(str)+X_test.Sex.astype(str)

#######################################################################
# mean encoding
#######################################################################
all_data = X_train.copy()
all_data['target'] = y_train.copy()

# quick and dirty mean encode
for col in categorical+text+generated+boolean:
    means = all_data.groupby(col).target.mean()
    X_train[col] = X_train[col].map(means)
    X_test[col] = X_test[col].map(means)

# fillnas
X_train.Age.fillna(-999, inplace=True)
X_train.Cabin.fillna(-999, inplace=True)
X_train.Embarked.fillna(-999, inplace=True)
X_test.Age.fillna(-999, inplace=True)
X_test.Cabin.fillna(-999, inplace=True)
X_test.Embarked.fillna(-999, inplace=True)
X_test.Fare.fillna(-999, inplace=True)
X_test.Name.fillna(-999, inplace=True)

#######################################################################
# base predictions
#######################################################################

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values

rf_params = {
    'n_jobs': -1,
    'n_estimators': 200,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

et_params = {
    'n_jobs': -1,
    'n_estimators':200,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

ada_params = {
    'n_estimators': 200,
    'learning_rate' : 0.75
}

gb_params = {
    'n_estimators': 200,
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

et_oof_train, et_oof_test = et.get_oof(X_train, y_train, X_test) # Extra Trees
rf_oof_train, rf_oof_test = rf.get_oof(X_train, y_train, X_test) # Random Forest
ada_oof_train, ada_oof_test = ada.get_oof(X_train, y_train, X_test) # AdaBoost
gb_oof_train, gb_oof_test = gb.get_oof(X_train, y_train, X_test) # Gradient Boost
svc_oof_train, svc_oof_test = svc.get_oof(X_train, y_train, X_test) # Support Vector Classifier

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(), 'ExtraTrees': et_oof_train.ravel(), 'AdaBoost': ada_oof_train.ravel(),  'GradientBoost': gb_oof_train.ravel(), 'SupportVector': svc_oof_train.ravel()
    })

sns.heatmap(base_predictions_train.corr())

#######################################################################
# second level learning
#######################################################################

X_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
X_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

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
 scale_pos_weight=1).fit(X_train, y_train)

predictions = gbm.predict(X_test)
results = pd.concat([ids, pd.Series(predictions, name='Survived')], axis=1)
results.to_csv('titanic2.csv', index=False)
