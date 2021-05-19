import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
from pprint import pprint
from sklearn.utils import resample
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier


def yesno_to_num(x):
    if x == 'No':
        return 0
    if x == 'Yes':
        return 1
    else:
        return x


# Import the protein dataset
protein = pd.read_csv("/Users/luzhuye/Desktop/protein_project/protein-localization/train.csv",
                      na_values="?",
                      header=None)
df = protein.iloc[0:861, 1:-2]
y = protein.iloc[0:861, -1]


# Preprocess the data
# - Drop all columns with only NaNs
df = df.dropna(axis=1, how='all').T.reset_index(drop=True).T

# - Fill NaN with medium for numeric columns/
#   the most frequent value for categorical columns
num_coln = df.select_dtypes(include=['number']).columns
df[num_coln] = df[num_coln].apply(lambda x: x.fillna(x.median(), inplace=True))
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

# - Encode yes/no to columns(Class, Complex, Phenotype, Motif)
df.iloc[:, 1:457] = df.iloc[:,1:457].apply(lambda x: x.apply(yesno_to_num))
df.iloc[:, -14:] = df.iloc[:,-14:].apply(lambda x: x.apply(yesno_to_num))

# - One-hot encoding to columns(Essential, Interaction, Chromosome)
encoded_essent = pd.get_dummies(df.iloc[:, 0], prefix='Essent')
df = df.drop(df.columns[0], axis=1).join(encoded_essent)
encoded_inter = pd.get_dummies(df.iloc[:, 457:-18], prefix='Inter')
df = df.drop(df.iloc[:, 457:-18], axis=1).join(encoded_inter)
encoded_chromo = pd.get_dummies(df.iloc[:, 443], prefix='Chromo')
df = df.drop(df.columns[442], axis=1).join(encoded_chromo)
df.drop(df.iloc[:, 442:456], axis=1, inplace=True)

# - Drop columns which contain only one distinct value
df.drop(columns=df.columns[df.nunique() == 1], inplace=True)
print(df.head())


# Label histogram
plt.hist(y, color='blue', edgecolor='black',
         bins=int(15/1))
plt.xlabel('class')
plt.ylabel('frequency')
plt.show()


# - Use Oversample to generate data for class 11, 12, 14
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=27)
train = pd.concat([X_train, y_train], axis=1)
class_11 = train[train.iloc[:, -1] == 11]
class_12 = train[train.iloc[:, -1] == 12]
class_14 = train[train.iloc[:, -1] == 14]
unsampled_11 = resample(class_11, replace=True, n_samples=270, random_state=27)
unsampled_12 = resample(class_12, replace=True, n_samples=270, random_state=27)
unsampled_14 = resample(class_14, replace=True, n_samples=270, random_state=27)
unsampled = pd.concat([train, unsampled_11, unsampled_12, unsampled_14])
X_train = unsampled.iloc[:, 0:-1]
y_train = unsampled.iloc[:, -1]

# - Use SMOTE to generate new and synthetic data for class 0-10
sm = SMOTE(random_state=27,
           k_neighbors=3,
           sampling_strategy={0: 271, 1: 271, 2: 271, 3: 271, 4: 271, 5: 271,
                              6: 271, 7: 271, 8: 271, 9: 271, 10: 271})
X_train, y_train = sm.fit_resample(np.array(X_train), y_train)
print(y_train.value_counts())

'''
# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
# Number of trees in random forest
n_estimators = [300, 500, 700]
# Number of features to consider at every split
max_features = ['sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(60, 100, num=5)]
# Minimum number of samples required to split a node
min_samples_split = [10, 12, 13]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 3, 4]
# Create the hyperparameter grid
hyperparam_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
pprint(hyperparam_grid)

# Instantiate a Base model and a Random search of parameters
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=hyperparam_grid, n_iter=150, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the model
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)

# Evaluate the RandomForest model
y_pred_train = rf_random.predict(X_train)
print('Training dataset accuracy score: ', accuracy_score(y_pred_train, y_train))
y_pred_test = rf_random.predict(np.array(X_test))
print('Testing dataset accuracy score: ', accuracy_score(y_pred_test, y_test))
'''

'''
# K-Nearest Neighbor Classifier
knn = KNeighborsClassifier()
# Tune Hyper-parameters
weights = ['uniform', 'distance']
n_neighbors = [2, 3, 4]
print(n_neighbors)
p = [1, 2, 3]
# Create the Hyper-parameter Grid
hyperparam_grid = {'weights': weights,
                   'n_neighbors': n_neighbors,
                   'p': p}
clf = GridSearchCV(knn, hyperparam_grid, cv=3)
# Fit the model
knn_model = clf.fit(X_train, y_train)
print(knn_model.best_params_)

# Evaluate the Knn model
y_pred_train = knn_model.predict(X_train)
print('Training dataset accuracy score: ', accuracy_score(y_pred_train, y_train))
y_pred_test = knn_model.predict(np.array(X_test))
print('Testing dataset accuracy score: ', accuracy_score(y_pred_test, y_test))
'''

'''
# Support Vector Machine
svm = SVC()
# Tune Hyper-parameters
C = [0.1, 1, 10, 100]
gamma = [1, 0.1, 0.001, 0.0001]
# Create the Hyper-parameter Grid
hyperparam_grid = {'C': C,
                   'gamma': gamma,
                   'kernel': ['rbf']}
clf = GridSearchCV(svm, hyperparam_grid, cv=3)
# Fit the model
svm_model = clf.fit(X_train, y_train)
print(svm_model.best_params_)

# Evaluate SVM model
y_pred_train = svm_model.predict(X_train)
print('Training dataset accuracy score: ', accuracy_score(y_pred_train, y_train))
y_pred_test = svm_model.predict(np.array(X_test))
print('Testing dataset accuracy score: ', accuracy_score(y_pred_test, y_test))
'''

'''
# Gradient Boosting Machines (takes super long to run, 
# answers: https://stats.stackexchange.com/questions/144897/gradientboostclassifiersklearn-takes-very-long-time-to-train)
gbc = GradientBoostingClassifier()
# Tune Hyper-parameters
learning_rate = [0.01, 0.1, 0.5, 1]
n_estimators = [100, 500, 800]
# Create the Hyper-parameter Grid
hyperparam_grid = {'learning_rate': learning_rate,
                   'n_estimators': n_estimators}
clf = RandomizedSearchCV(estimator=gbc,
                         param_distributions=hyperparam_grid, n_iter=2, cv=3, verbose=2,
                         random_state=42, n_jobs=-1)
# Fit the model
gbc_model = clf.fit(X_train, y_train)
print(gbc_model.best_params_)

# Evaluate GBC model
y_pred_train = gbc_model.predict(X_train)
print('Training dataset accuracy score: ', accuracy_score(y_pred_train, y_train))
y_pred_test = gbc_model.predict(np.array(X_test))
print('Testing dataset accuracy score: ', accuracy_score(y_pred_test, y_test))



# XGBoost
xgb = XGBClassifier()
# Tune Hyper-parameters
eta = [0.01, 0.1, 0.5, 1]
max_depth = [2, 4, 6]
gamma = [0.5, 2.5, 5]
lambdas = [0, 0.5, 1]
# Create the Hyper-parameter Grid
hyperparam_grid = {'eta': eta,
                   'max_depth': max_depth,
                   'gamma': gamma,
                   'lambda': lambdas}
clf = RandomizedSearchCV(estimator=xgb,
                         param_distributions=hyperparam_grid, n_iter=2, cv=3, verbose=2,
                         random_state=42, n_jobs=-1)
# Fit the model
xgb_model = clf.fit(X_train, y_train)
print(xgb_model.best_params_)

# Evaluate XGB model
y_pred_train = xgb_model.predict(X_train)
print('Training dataset accuracy score: ', accuracy_score(y_pred_train, y_train))
y_pred_test = xgb_model.predict(np.array(X_test))
print('Testing dataset accuracy score: ', accuracy_score(y_pred_test, y_test))
'''


# Stacking
# Define Base Learners
models = dict()
models['lr'] = LogisticRegression()
models['bayes'] = GaussianNB()
models['rfc'] = RandomForestClassifier()
models['knn'] = KNeighborsClassifier()
models['svm'] = SVC()

# Define Meta Model
lr = LogisticRegression()
models['stackingclf'] = StackingClassifier(estimators=[('lr', models['lr']), ('bayes', models['bayes']),
                                                       ('rfc', models['rfc']), ('knn', models['knn']),
                                                       ('svm', models['svm'])],
                                           final_estimator=lr)

# Evaluate models with 3-fold cross validation
pred, names = list(), list()
for name, model in models.items():
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, np.array(df), y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    pred.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
plt.boxplot(pred, labels=names, showmeans=True)
plt.show()

