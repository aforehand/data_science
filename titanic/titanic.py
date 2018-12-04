import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')
y = train_df.Survived
X = train_df.loc[:,'Pclass':]

X_eng = X.copy()
X_eng.Age.replace(np.nan, -1, inplace=True)
X_eng.Cabin.replace(np.nan, '', inplace=True)
X_eng.Embarked.replace(np.nan, '', inplace=True)

# retain only honorifics in names
X_eng.Name.replace('^(.+), ', '', regex=True, inplace=True)
X_eng.Name.replace('\..*', '.', regex=True, inplace=True)
X_eng.Name = X_eng.Name.str.strip()

# reduce cabin to deck
X_eng.Cabin.replace('[0-9]+.*', '', regex=True, inplace=True)

# encoding categorical features
X_eng_e = X_eng.copy()
categorical = ['Sex', 'Name', 'Ticket', 'Cabin', 'Embarked']
X_eng_e.loc[:,categorical] = X_eng.loc[:,categorical].astype(np.str).apply(LabelEncoder().fit_transform)

# adding some features
X_eng_e['FamilySize'] = X_eng_e.SibSp + X_eng_e.Parch
X_eng_e['SexClass'] = (X_eng_e.Sex + 1) * X_eng_e.Pclass
X_eng_e['AgeSex'] = (X_eng_e.Sex + 1) * X_eng_e.Age

gbc = GradientBoostingClassifier(max_depth=5, random_state=0)
scores = cross_val_score(gbc, X_eng_e, y)
scores.mean()
# mean = 0.846; sd = 0.024

#######################################################################
# test set
test_df = pd.read_csv('test.csv')
X_test = test_df.loc[:,'Pclass':]
ids = test_df.PassengerId

# retain only honorifics in names
X_test.Name.replace('^(.+), ', '', regex=True, inplace=True)
X_test.Name.replace('\..*', '.', regex=True, inplace=True)
X_test.Name = X_test.Name.str.strip()

# reduce cabin to deck
X_test.Cabin.replace('[0-9]+.*', '', regex=True, inplace=True)

# encoding categorical features
X_test_e = X_test.copy()
X_test_e.loc[:,categorical] = X_test.loc[:,categorical].astype(np.str).apply(LabelEncoder().fit_transform)

# adding some features
X_test_e['FamilySize'] = X_test_e.SibSp + X_test_e.Parch
X_test_e['SexClass'] = (X_test_e.Sex + 1) * X_test_e.Pclass
X_test_e['AgeSex'] = (X_test_e.Sex + 1) * X_test_e.Age

gbc = GradientBoostingClassifier(max_depth=5, random_state=0).fit(X_eng_e, y)
survival_predictions = gbc.predict(X_test_e)
results = pd.concat([ids, pd.Series(survival_predictions, name='Survived')], axis=1)
results.to_csv('titanic.csv', index=False)
