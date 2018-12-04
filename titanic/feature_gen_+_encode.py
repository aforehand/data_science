#######################################################################
# next steps
#######################################################################

# implement a good cross-validation strategy in training the models to find
# optimal parameter values
# introduce a greater variety of base models. the more uncorrelated the base
# models, the better the results

# possible feature changes
# look at each feature more carefully to see if there are more appropriate
# values to for fillna
# there are many ages missing, look at average/most common age for titles to
# guess missing values
# create new features instead of replacing original ones for wacky encodings
# compare performance for mean- vs label-encoded
# look into which models require regularization, maybe good to do?
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('train.csv')

numeric = ['Age', 'Fare', 'Age_Buckets', 'Fare_Buckets']
count = ['SibSp', 'Parch', 'FamSize']
categorical = ['Sex', 'Cabin', 'Embarked']
ordinal = ['Pclass']
text = ['Name']
generated = ['AgeClass', 'AgeSex', 'FamClass', 'NameSex', 'SexMarried', 'AgeSexMarried', 'SexClass', 'SexCabin']
boolean = ['IsAlone', 'HasCabin', 'IsMarried']

#######################################################################

train_df.drop('Ticket', axis=1, inplace=True)
train_df.drop('PassengerId', axis=1, inplace=True)

# more age categories
train_df.Age.fillna(-999, inplace=True)
train_df['Age_Buckets'] = pd.cut(train_df.Age, [-9999,0,2,12,19,30,50,80], labels=[-999,0,1,2,3,4,5])
train_df['Fare_Buckets'] = pd.qcut(train_df.Fare, 4, labels=False)

train_df.Cabin.replace('[0-9]+.*', '', regex=True, inplace=True)

# retain only honorifics in names, grouping uncommon titles
train_df.Name.replace('^(.+), ', '', regex=True, inplace=True)
train_df.Name.replace('\..*', '.', regex=True, inplace=True)
names = {'Mr.': 'Mr', 'Mrs.': 'Mrs', 'Miss.': 'Miss', 'Master.': 'Master', 'Don.': 'Uncommon', 'Rev.': 'Uncommon', 'Dr.': 'Uncommon', 'Mme.': 'Uncommon','Ms.': 'Uncommon', 'Major.': 'Uncommon', 'Lady.': 'Uncommon', 'Sir.': 'Uncommon', 'Mlle.': 'Uncommon', 'Col.': 'Uncommon', 'Capt.': 'Uncommon', 'the Countess.': 'Uncommon', 'Jonkheer.': 'Uncommon'}
train_df['Name'] = train_df.Name.map(names)

train_df['FamSize'] = train_df.SibSp+train_df.Parch
train_df['IsAlone'] = (train_df.FamSize == 0.0)
#######################################################################

# fill nans
train_df.isnull().sum()
# Age: 177
# Cabin: 687
# Embarked: 2

train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df['Cabin'] = train_df['Cabin'].fillna('None')

# temp fill for fiddling
no_age = train_df[train_df.Age_Buckets==-999]
# find how many nans each name has
no_age.groupby('Name')['Name'].count()
# Master        4
# Miss         36
# Mr          119
# Mrs          17
# Uncommon      1

# looked at bar plots for various factors and assigned values based on
# what looked the most similar
train_df.loc[train_df.Name=='Master'] = train_df[train_df.Name=='Master'].replace(-999, 1)

train_df.loc[train_df.Name=='Uncommon'] = train_df[train_df.Name=='Uncommon'].replace(-999, 4)

train_df.loc[train_df.Name=='Mr'] = train_df[train_df.Name=='Mr'].replace(-999,2)

train_df.loc[(train_df.Name=='Mrs')&(train_df.IsAlone==True)] = train_df[(train_df.Name=='Mrs')&(train_df.IsAlone==True)].replace(-999,5)
train_df.loc[train_df.Name=='Mrs'] = train_df[train_df.Name=='Mrs'].replace(-999,2)

train_df.loc[(train_df.Name=='Miss')&(train_df.Parch==1)] = train_df[(train_df.Name=='Miss')&(train_df.Parch==1)].replace(-999,1)
train_df.loc[(train_df.Name=='Miss')&(train_df.SibSp==2)] = train_df[(train_df.Name=='Miss')&(train_df.SibSp==2)].replace(-999,1)
train_df.loc[train_df.Name=='Miss'] = train_df[train_df.Name=='Miss'].replace(-999,2)

#######################################################################

# feature generation
# analyze feature interactions with e.g.
# sns.barplot(x='Cabin', y='Survived', hue='Sex', data=train_df)

train_df['AgeClass'] = train_df.Age_Buckets.astype(str)+train_df.Pclass.astype(str)
train_df['HasCabin'] = ~(train_df.Cabin=='None')
train_df['AgeSex'] = train_df.Age_Buckets.astype(str)+train_df.Sex.astype(str)
train_df['FamClass'] = train_df.FamSize.astype(str)+train_df.Pclass.astype(str)
train_df['NameSex'] = train_df.Name.astype(str)+train_df.Sex.astype(str)
train_df['IsMarried'] = ((train_df.Name=='Mrs') | ((train_df.Sex=='male') & (train_df.Age_Buckets > 2) & (train_df.SibSp==1)))
train_df['SexMarried'] = train_df.Sex.astype(str)+train_df.IsMarried.astype(str)
train_df['AgeSexMarried'] = train_df.AgeSex.astype(str)+train_df.IsMarried.astype(str)
train_df['SexClass'] = train_df.Sex.astype(str)+train_df.Pclass.astype(str)
train_df['SexCabin'] = train_df.Sex.astype(str)+train_df.HasCabin.astype(str)
#######################################################################

train_df['Age'] = train_df.Age_Buckets.values
train_df.drop('Age_Buckets', axis=1, inplace=True)
train_df['Age'] = train_df.Age.astype(int)
train_df['Fare'] = train_df.Fare_Buckets.values
train_df.drop('Fare_Buckets', axis=1, inplace=True)
#######################################################################

# label encoding
train_labels = train_df.copy()
le = LabelEncoder()
for col in categorical+text+generated+boolean:
    train_labels.loc[:,col] = le.fit_transform(train_labels.loc[:,col].astype(str))

X = train_labels.loc[:,'Pclass':]
y = train_labels['Survived']
forest = RandomForestClassifier(random_state=0)
scores = cross_val_score(forest, X, y)
scores.mean()
# 0.809

# ensemble score: 0.839

# mean encoding
train_means = train_df.copy()
for col in categorical+text+generated+boolean:
    means = train_means.iloc[250:,:].groupby(col).Survived.mean()
    train_means[col] = train_means[col].map(means)

train_means.isnull().sum()
# ugh
train_means.Cabin.fillna(train_means.Survived, inplace=True)

# forest score: train-0.912; val-0.820


train_mix = train_df.copy()
le = LabelEncoder()
for col in categorical+text+boolean:
    train_mix.loc[:,col] = le.fit_transform(train_mix.loc[:,col].astype(str))

for col in generated:
    means = train_mix.iloc[250:,:].groupby(col).Survived.mean()
    train_mix[col] = train_mix[col].map(means)

# forest mean: 0.814
# ensemble: 0.825
#######################################################################
# test
#######################################################################
test_df = pd.read_csv('test.csv')

test_df.drop('Ticket', axis=1, inplace=True)

# more age categories
test_df.Age.fillna(-999, inplace=True)
test_df['Age_Buckets'] = pd.cut(test_df.Age, [-9999,0,2,12,19,30,50,80], labels=[-999,0,1,2,3,4,5])
test_df['Fare_Buckets'] = pd.qcut(test_df.Fare, 4, labels=False)

test_df.Cabin.replace('[0-9]+.*', '', regex=True, inplace=True)

# retain only honorifics in names, grouping uncommon titles
test_df.Name.replace('^(.+), ', '', regex=True, inplace=True)
test_df.Name.replace('\..*', '.', regex=True, inplace=True)
names = {'Mr.': 'Mr', 'Mrs.': 'Mrs', 'Miss.': 'Miss', 'Master.': 'Master', 'Don.': 'Uncommon', 'Rev.': 'Uncommon', 'Dr.': 'Uncommon', 'Mme.': 'Uncommon','Ms.': 'Uncommon', 'Major.': 'Uncommon', 'Lady.': 'Uncommon', 'Sir.': 'Uncommon', 'Mlle.': 'Uncommon', 'Col.': 'Uncommon', 'Capt.': 'Uncommon', 'the Countess.': 'Uncommon', 'Jonkheer.': 'Uncommon'}
test_df['Name'] = test_df.Name.map(names)

test_df['FamSize'] = test_df.SibSp+test_df.Parch
test_df['IsAlone'] = (test_df.FamSize == 0.0)
#######################################################################

test_df['Embarked'] = test_df['Embarked'].fillna('S')
test_df['Cabin'] = test_df['Cabin'].fillna('None')

test_df.loc[test_df.Name=='Master'] = test_df[test_df.Name=='Master'].replace(-999, 1)

test_df.loc[test_df.Name=='Uncommon'] = test_df[test_df.Name=='Uncommon'].replace(-999, 4)

test_df.loc[test_df.Name=='Mr'] = test_df[test_df.Name=='Mr'].replace(-999,2)

test_df.loc[(test_df.Name=='Mrs')&(test_df.IsAlone==True)] = test_df[(test_df.Name=='Mrs')&(test_df.IsAlone==True)].replace(-999,5)
test_df.loc[test_df.Name=='Mrs'] = test_df[test_df.Name=='Mrs'].replace(-999,2)

test_df.loc[(test_df.Name=='Miss')&(test_df.Parch==1)] = test_df[(test_df.Name=='Miss')&(test_df.Parch==1)].replace(-999,1)
test_df.loc[(test_df.Name=='Miss')&(test_df.SibSp==2)] = test_df[(test_df.Name=='Miss')&(test_df.SibSp==2)].replace(-999,1)
test_df.loc[test_df.Name=='Miss'] = test_df[test_df.Name=='Miss'].replace(-999,2)
#######################################################################

test_df['AgeClass'] = test_df.Age_Buckets.astype(str)+test_df.Pclass.astype(str)
test_df['HasCabin'] = ~(test_df.Cabin=='None')
test_df['AgeSex'] = test_df.Age_Buckets.astype(str)+test_df.Sex.astype(str)
test_df['FamClass'] = test_df.FamSize.astype(str)+test_df.Pclass.astype(str)
test_df['NameSex'] = test_df.Name.astype(str)+test_df.Sex.astype(str)
test_df['IsMarried'] = ((test_df.Name=='Mrs') | ((test_df.Sex=='male') & (test_df.Age_Buckets > 2) & (test_df.SibSp==1)))
test_df['SexMarried'] = test_df.Sex.astype(str)+test_df.IsMarried.astype(str)
test_df['AgeSexMarried'] = test_df.AgeSex.astype(str)+test_df.IsMarried.astype(str)
test_df['SexClass'] = test_df.Sex.astype(str)+test_df.Pclass.astype(str)
test_df['SexCabin'] = test_df.Sex.astype(str)+test_df.HasCabin.astype(str)
#######################################################################

test_df['Age'] = test_df.Age_Buckets.values
test_df.drop('Age_Buckets', axis=1, inplace=True)
test_df['Age'] = test_df.Age.astype(int)
test_df['Fare'] = test_df.Fare_Buckets.values
test_df.drop('Fare_Buckets', axis=1, inplace=True)
#######################################################################
test_df.isnull().sum()
# name: 1
# fare: 1
test_df.fillna(-999, inplace=True)
# fix remaining nans
# was alone and middle aged so probably miss
test_df.Name.replace(-999, 'Miss', inplace=True)
test_df.NameSex.replace('nanfemale', 'Missfemale', inplace=True)
# 3rd class and no cabin
test_df.Fare.replace(-999,0,inplace=True)
#######################################################################

train_mix = train_df.copy()
test_mix = test_df.copy()
le = LabelEncoder()
for col in categorical+text+boolean:
    train_mix.loc[:,col] = le.fit_transform(train_mix.loc[:,col].astype(str))
    test_mix.loc[:,col] = le.transform(test_mix.loc[:,col].astype(str))

for col in generated:
    means = train_mix.iloc[250:,:].groupby(col).Survived.mean()
    train_mix[col] = train_mix[col].map(means)
    test_mix[col] = test_mix[col].map(means)
