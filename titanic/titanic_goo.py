import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('train.csv')
y = train_df.Survived
X = train_df.loc[:,'Pclass':]
ids = train_df.PassengerId

# sets for baseline performance
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

#encoded sets for algorithms that need thinat
X_encode = X.copy()
categorical = ['Sex', 'Name', 'Ticket', 'Cabin', 'Embarked']
X_encode.loc[:,categorical] = X.loc[:,categorical].astype(np.str).apply(LabelEncoder().fit_transform)

X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(X_encode, y, random_state=0)
#######################################################################
# dummy
#######################################################################
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(random_state=0).fit(X_train_e, y_train_e)
dummy.score(X_train_e, y_train_e)
# 0.501
dummy.score(X_val_e, y_val_e)
# 0.538
#######################################################################
# decision tree
#######################################################################
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0).fit(X_train_e, y_train_e)
tree.score(X_train_e, y_train_e)
# 1.0
tree.score(X_val_e, y_val_e)
# 0.785
# way overfit, so pruning
tree = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train_e, y_train_e)
tree.score(X_train_e, y_train_e
# 0.855
tree.score(X_val_e, y_val_e)
# 0.816
#######################################################################
# random forest
#######################################################################
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=0).fit(X_train_e, y_train_e)
forest.score(X_train_e, y_train_e)
# 0.981
forest.score(X_val_e, y_val_e)
# 0.857
# a bit overfit, but tweaking does not improve validation score
#######################################################################
# KNN
#######################################################################
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier().fit(X_train_e, y_train_e)
knn.score(X_train_e, y_train_e)
# 0.753
knn.score(X_val_e, y_val_e)
# 0.699
# meh
#######################################################################
# SVM
#######################################################################
from sklearn.smv import SVC
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_e)
X_val_scaled = scaler.transform(X_val_e)

svc = SVC(random_state=0).fit(X_train_e, y_train_e)
svc.score(X_train_e, y_train_e)
# 0.789
svc.score(X_val_e, y_val_e)
# 0.780
svc = SVC(gamma=1, C=10, random_state=0).fit(X_train_scaled, y_train)
svc.score(X_train_scaled, y_train)
# 0.874
svc.score(X_val_scaled, y_val)
# 0.830
#######################################################################
# neural net
#######################################################################
from sklearn.neral_network import MLPClassifier

nn = MLPClassifier(activation='tanh', random_state=0).fit(X_train_e, y_train_e)
nn.score(X_train_e, y_train_e)
# 0.793
nn.score(X_val_e, y_val_e)
# 0.758
#######################################################################
# logistic regression
#######################################################################
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)
lr.score(X_train_scaled, y_train)
# 0.789
lr.score(X_val_scaled, y_val)
# 0.794
#######################################################################
# SGD
#######################################################################
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(random_state=0).fit(X_train_scaled, y_train)
sgd.score(X_train_scaled, y_train)
# 0.793
sgd.score(X_val_scaled, y_val)
# 0.776
#######################################################################
# random forest performed best, gonna do some feature engineering
# then check out some ensembles
#######################################################################
X_eng = X.copy()
X_eng.Age.replace(np.nan, -1, inplace=True)
X_eng.Cabin.replace(np.nan, '', inplace=True)
X_eng.Embarked.replace(np.nan, '', inplace=True)

# retain only honorifics in names
X_eng.Name.replace('^(.+), ', '', regex=True, inplace=True)
X_eng.Name.replace('\..*', '.', regex=True, inplace=True)
#not encoded
X_train_eng, X_val_eng, y_train_eng, y_val_eng = train_test_split(X_eng, y, random_state=0)

X_eng_e = X_eng.copy()
X_eng_e.loc[:,categorical] = X_eng.loc[:,categorical].astype(np.str).apply(LabelEncoder().fit_transform)
#encoded
X_train_eng_e, X_val_eng_e, y_train_eng_e, y_val_eng_e = train_test_split(X_eng_e, y, random_state=0)

forest = RandomForestClassifier(random_state=0).fit(X_train_eng_e, y_train_eng_e)
forest.score(X_train_eng_e, y_train_eng_e)
# 0.978
forest.score(X_val_eng_e, y_val_eng_e)
# 0.865

# get size of whole family
X_eng['FamilySize'] = X_eng.SibSp+X_eng.Parch
# train: 0.979
# test: 0.834

# reduce cabin to deck
 X_eng.Cabin.replace('[0-9]+.*', '', regex=True, inplace=True)
# train: 0.975
# val: 0.852
from sklearn.model_selection import cross_val_score

forest = RandomForestClassifier(random_state=0)
scores = cross_val_score(forest, X_eng_e, y)
scores.mean()
# 0.813
#######################################################################
# extremely randomized trees
from sklearn.ensemble import ExtraTreesClassifier

extra = ExtraTreesClassifier(random_state=0)
scores = cross_val_score(extra, X_eng_e, y)
scores.mean()
# 0.796
extra = ExtraTreesClassifier(max_features=4, random_state=0)
scores = cross_val_score(extra, X_eng_e, y)
scores.mean()
# 0.802
#######################################################################
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(random_state=0)
scores = cross_val_score(ada, X_eng_e, y)
scores.mean()
# 0.817
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=0)
scores = cross_val_score(ada, X_eng_e, y)
scores.mean()
# 0.826
ada = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=6, max_depth=1), n_estimators=100, random_state=0)
scores = cross_val_score(ada, X_eng_e, y)
scores.mean()
# 0.828
#######################################################################
# gradient boosting classifier
# note: gets slow fast with more classes
from sklearn.enseomble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)
scores = cross_val_score(gbc, X_eng_e, y)
scores.mean()
# 0.836
gbc = GradientBoostingClassifier(max_depth=5, random_state=0)
scores = cross_val_score(gbc, X_eng_e, y)
scores.mean()
# 0.838
#######################################################################
# more engineering, all scores are for the above classifier
#######################################################################
# there were weird whitespace characters increasing noise
X_eng.Name = X_eng.Name.str.strip()
X_eng_e['SexClass'] = (X_eng_e.Sex + 1) * X_eng_e.Pclass
# 0.841
X_eng_e['AgeSex'] = (X_eng_e.Sex +1) * X_eng_e.Age
# 0.846
