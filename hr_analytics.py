# -*- coding: utf-8 -*-
"""FML Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n2IcF8SsAbhuk64bkPoPgHFlS_MIfKzA

#### https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

### target: 0 – Not looking for job change, 1 – Looking for a job change

# Import the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_roc_curve
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)

# Import the data
train_dataset = pd.read_csv('aug_train.csv')

#               Visualize data

# Different plots
def bar_plot(data, col, title=None, display_pct=False, hue=None):
    ax = sns.countplot(data=data, x=col, order=data[col].value_counts().index, hue=hue)
    if title is None:
        plt.title('Distribution of ' + col)
    else:
        plt.title(title)
    plt.xlabel(col)
    if display_pct:
        labels = (data[col].value_counts())
        for i, v in enumerate(labels):
            ax.text(i, v + 10, str(v), horizontalalignment='center', size=14)
    plt.show()


def box_plot(data, col1, col2, title=None, orient='h'):
    sns.boxplot(data=data, x=col1, y=col2, orient=orient)
    plt.title(title)
    plt.show()


def histogram_plot(data, col, title=None, hue=None):
    sns.displot(data, x=col, hue=hue, element="step")
    # plt.title(title)
    plt.show()


def pie_plot(data, col, title=None):
    plt.figure(figsize=(6, 6))
    piedata = data[col].value_counts()
    plt.pie(x=piedata, autopct="%.1f%%", labels=piedata.index)
    plt.title(title)
    plt.show()


def pair_plot(data):
    sns.pairplot(data)
    plt.show()


histogram_plot(train_dataset, 'city_development_index', hue='target', title='City development index')
# looks like people from developed cities are less likely to look for a job change


# Gender distribution
bar_plot(train_dataset, 'gender', title='Gender distribution', hue='target')


# Relevant experience: there are employees with more than 20 years of experience in distribution of
# both, employees with relevant experience and no relevant experience
bar_plot(train_dataset, 'relevent_experience', hue='target')
no_relevant_exp = train_dataset[train_dataset['relevent_experience'] == 'No relevent experience']
bar_plot(no_relevant_exp, 'experience', 'No relevant experience')
relevant_exp = train_dataset[train_dataset['relevent_experience'] != 'No relevent experience']
bar_plot(relevant_exp, 'experience', 'Relevant experience')

# visualize enrolled in university
looking_for_job = train_dataset[train_dataset['target'] == 1]
not_looking = train_dataset[train_dataset['target'] == 0]
pie_plot(looking_for_job, 'enrolled_university', 'Looking for jobs')
pie_plot(not_looking, 'enrolled_university', 'Not looking for jobs')


# education level
pie_plot(looking_for_job, 'education_level', 'Looking for jobs')
pie_plot(not_looking, 'education_level', 'Not looking for jobs')


# Visualize company size
bar_plot(train_dataset, 'company_size', hue='target')

# Visualize for last new job
bar_plot(train_dataset, 'last_new_job', hue='target')


# Pair plot - everything with everything
pair_plot(train_dataset)


#           Cleanup data

# Quick description of data
print(train_dataset.head())
print(train_dataset.info())
print(train_dataset.describe(include='all'))


# Is there any missing data?
print(train_dataset.isnull().mean() * 100)
print(train_dataset.isnull().sum())


# Dropping what is under 3%
train_dataset.dropna(subset=['enrolled_university', 'education_level', 'experience', 'last_new_job'], inplace=True)
print(train_dataset.isnull().mean() * 100)


# Dropping "enrolee_id" as it is an index
train_dataset.drop('enrollee_id', axis=1, inplace=True)
print(train_dataset.head())


# City and city_development_index are having similar information, they are strictly linked together so dropping 'city'
train_dataset.drop('city', axis=1, inplace=True)
print(train_dataset.head())


# Splitting into Features and Labels"""
X = train_dataset.drop('target', axis=1)
y = train_dataset['target']


# Fixing the Gender - replacing NULL with 'Female' (index[1])
X.loc[X['gender'].isnull(), 'gender'] = X['gender'].value_counts().index[1]
print(X.head())


# Fixing 'major_discipline'
X.loc[X['major_discipline'].isnull(), 'major_discipline'] = X['major_discipline'].value_counts().index[1]
print(X.head())


# Fixing 'company_size' and 'company_type'
X.loc[X['company_size'].isnull(), 'company_size'] = X['company_size'].value_counts().index[0]
X.loc[X['company_type'].isnull(), 'company_type'] = X['company_type'].value_counts().index[0]
print(X.head())
print(X.isnull().mean() * 100)


#       Encoding features

#   Part 1
X.loc[X['relevent_experience'] == 'No relevent experience', 'relevent_experience'] = 0
X.loc[X['relevent_experience'] == 'Has relevent experience', 'relevent_experience'] = 1

X.loc[X['education_level'] == 'Primary School', 'education_level'] = 0
X.loc[X['education_level'] == 'High School', 'education_level'] = 1
X.loc[X['education_level'] == 'Graduate', 'education_level'] = 2
X.loc[X['education_level'] == 'Masters', 'education_level'] = 3
X.loc[X['education_level'] == 'Phd', 'education_level'] = 4

X.loc[X['company_size'] == '0', 'company_size'] = 0
X.loc[X['company_size'] == '<10', 'company_size'] = 1
X.loc[X['company_size'] == '10/49', 'company_size'] = 2
X.loc[X['company_size'] == '50-99', 'company_size'] = 3
X.loc[X['company_size'] == '100-500', 'company_size'] = 4
X.loc[X['company_size'] == '500-999', 'company_size'] = 5
X.loc[X['company_size'] == '1000-4999', 'company_size'] = 6
X.loc[X['company_size'] == '5000-9999', 'company_size'] = 7
X.loc[X['company_size'] == '10000+', 'company_size'] = 8

X.loc[X['experience'] == '>20', 'experience'] = 21
X.loc[X['experience'] == '<1', 'experience'] = 0
X.loc[X['last_new_job'] == 'never', 'last_new_job'] = 0
X.loc[X['last_new_job'] == '>4', 'last_new_job'] = 5

X['experience'] = X['experience'].astype(int)
X['last_new_job'] = X['last_new_job'].astype(int)
X['city_development_index'] = X['city_development_index'].astype(float)
X['city_development_index'] = X['city_development_index'].apply(lambda x: x * 1000)
X['training_hours'] = X['training_hours'].astype(int)

print(X.head())


#   Part 2

num_attribs = ['city_development_index', 'education_level', 'experience', 'company_size',
               'last_new_job', 'training_hours']
cat_attribs = ['gender', 'enrolled_university', 'major_discipline', 'company_type']
preprocessing = ColumnTransformer([
    ("num", StandardScaler(), num_attribs),
    ("cat", OneHotEncoder(drop='first', sparse=False), cat_attribs)
], remainder="passthrough")

hr_processed = preprocessing.fit_transform(X)
print(hr_processed)


#       Fitting data

#   Model function
ax = plt.gca()


def Model(model, X_train, X_test, y_train, y_test, title):
    # https://scikit-learn.org/stable/modules/cross_validation.html
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    val_score = cross_val_score(model, X_test, y_test)

    print(title + ' - training set - accuracy score: ', accuracy_score(y_train, y_train_pred))
    print(title + ' - test set - accuracy score: ', accuracy_score(y_test, y_test_pred))
    print(title + ' - training set - confusion matrix: \n', confusion_matrix(y_train, y_train_pred))
    print(title + ' - test set - confusion matrix: \n', confusion_matrix(y_test, y_test_pred))
    print(title + " - %0.2f accuracy with a standard deviation of %0.2f" % (val_score.mean(), val_score.std()))
    plot_roc_curve(model, X_test, y_test, ax=ax, alpha=0.8)
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)


# Split data
X_train, X_test, y_train, y_test = train_test_split(hr_processed, y, test_size=0.2, shuffle=True, stratify=y)


# Logistic regression
# Tuning parameters


LR = LogisticRegression()
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100, 800, 1000)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
grid = GridSearchCV(LR, param_grid=param_grid, refit=True, verbose=3)
# fitting the model for grid search
grid.fit(X_train, y_train)
grid.best_params_
# summarize
print('Mean Accuracy: %.3f' % grid.best_score_)
print('Config: %s' % grid.best_params_)

Model(LogisticRegression(C=0.1, max_iter=100, solver='liblinear'), X_train, X_test, y_train, y_test,
      "Logistic regression")

"""### Linear SVM

#### Tuning parameters
"""

param_grid = {'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01],
              'kernel': ['linear']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
print('Mean Accuracy: %.3f' % grid.best_score_)
print('Config: %s' % grid.best_params_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

Model(SVC(kernel='linear', C=0.1, gamma=1), X_train, X_test, y_train, y_test, "SVM model")

"""#### K-Nearest Neighbors (K-NN)

##### Tuning the parameters
"""

param_grid = {'weights': ['uniform', 'distance'],
              'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
              'metric': ['minkowski', 'euclidian', 'manhatan']
              }
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3)
# fitting the model for grid search
grid.fit(X_train, y_train)
grid.best_params_
# summarize
print('Mean Accuracy: %.3f' % grid.best_score_)
print('Config: %s' % grid.best_params_)

Model(KNeighborsClassifier(n_neighbors=15, metric='minkowski', weights='uniform'), X_train, X_test, y_train, y_test,
      "K nearest neighbor")

"""#### Kernek-SVM

### Tuning the parameters
"""

param_grid = {'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01],
              'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, )
# fitting the model for grid search
grid.fit(X_train, y_train)
grid.best_params_
# summarize
print('Mean Accuracy: %.3f' % grid.best_score_)
print('Config: %s' % grid.best_params_)

Model(SVC(kernel='rbf', C=1, gamma=0.1), X_train, X_test, y_train, y_test, "Kernel SVM")
plt.show()