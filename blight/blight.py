import plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

train_df = pd.read_csv('train.csv', encoding = "ISO-8859-1")
test_df = pd.read_csv('test.csv')

# delete rows that do not have target data
train_df.dropna(subset=['compliance'], inplace=True)
train_df.reset_index(drop=True)

train_features = train_df.columns
test_features = test_df.columns

# remove features from training data that are not available for testing
X = train_df.drop(np.setdiff1d(train_features, test_features), axis=1)

continuous = ['admin_fee', 'clean_up_cost', 'discount_amount', 'fine_amount', 'judgment_amount', 'late_fee', 'state_fee']
categorical = list(np.setdiff1d(X.columns, continuous))


# encode categorical features
X_test = test_df
le = LabelEncoder().fit(np.union1d(X.loc[:,categorical].astype(np.str), X_test.loc[:,categorical].astype(np.str)))

X.loc[:,categorical] = X.loc[:,categorical].astype(np.str).apply(le.transform)

y_all = train_df.loc[:,np.setdiff1d(train_features, test_features)]
y = y_all.loc[:,'compliance']

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

# normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# least squares with l2 regularization
linridge = Ridge(alpha=1, random_state=0).fit(X_train_scaled, y_train)
y_linridge_predicted = linridge.predict(X_val_scaled)
plots.roc(y_val, y_linridge_predicted, title='ridge regression')
# classes are imbalanced, so using ROC AUC to score
roc_auc_score(y_val, y_linridge_predicted)
# 0.71

# apply polynomial regression for better ROC AUC score
# get polynomial and interaction features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_val_poly = poly.transform(X_val_scaled)

linridge = Ridge(alpha=1, random_state=0).fit(X_train_poly, y_train)
y_linridge_predicted = linridge.predict(X_val_poly)
plots.roc(y_val_poly, y_linridge_predicted, title='ridge regression with polynomial features')
roc_auc_score(y_val_poly, y_linridge_predicted)
# 0.80
#######################################################################
# evaluation
y_linridge_bool = (y_linridge_predicted > 0.5)
confusion_matrix(y_val_poly, y_linridge_bool)
# [[37056,    22],
#  [ 2650,   242]]
 print(classification_report(y_val, y_linridge_bool))
#              precision    recall  f1-score   support
#
#         0.0       0.93      1.00      0.97     37078
#         1.0       0.92      0.08      0.15      2892
#
# avg / total       0.93      0.93      0.91     39970

# determine important features
all_features = poly.get_feature_names(input_features=X.columns)
coefficients, features = zip(*sorted(zip(linridge.coef_, all_features), key=lambda t: abs(t[0]), reverse=True))
list(zip(features, coefficients))[:20]
# [('discount_amount^2', -4.4786455544462074),
#  ('disposition late_fee', -3.9927655024833775),
#  ('discount_amount', 1.7686482435321615),
#  ('country discount_amount', 1.7686482435321487),
#  ('non_us_str_code discount_amount', 1.7686482435321473),
#  ('violation_code discount_amount', -1.7324527978739599),
#  ('violation_code late_fee', 1.6787375559814457),
#  ('fine_amount discount_amount', -1.5675617435223761),
#  ('disposition fine_amount', 1.5455397917648008),
#  ('discount_amount judgment_amount', -1.4163697178440819),
#  ('fine_amount^2', -1.2226658663128467),
#  ('fine_amount late_fee', 1.1156178460731814),
#  ('late_fee^2', 1.1156178460731496),
#  ('late_fee judgment_amount', 1.1110669097667651),
#  ('disposition judgment_amount', 1.0380307422111044),
#  ('fine_amount judgment_amount', -1.0064016725706537),
#  ('disposition^2', 0.95551311875859091),
#  ('city discount_amount', 0.92753549696612558),
#  ('violator_name discount_amount', 0.84045825963066134),
#  ('judgment_amount^2', -0.81188363927148932)]
