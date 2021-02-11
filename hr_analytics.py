import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Dataset from https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists?select=sample_submission.csv

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


def visualize_data(train_dataset_local):
    # looks like people from developed cities are less likely to look for a job change
    histogram_plot(train_dataset_local, 'city_development_index', hue='target', title='City development index')

    # visualize gender distribution
    bar_plot(train_dataset_local, 'gender', title='Gender distribution', hue='target')

    # visualize relevant experience
    bar_plot(train_dataset_local, 'relevent_experience', hue='target')
    no_relevant_exp = train_dataset_local[train_dataset_local['relevent_experience'] == 'No relevent experience']
    bar_plot(no_relevant_exp, 'experience', 'No relevant experience')
    relevant_exp = train_dataset_local[train_dataset_local['relevent_experience'] != 'No relevent experience']
    bar_plot(relevant_exp, 'experience', 'Relevant experience')
    # there are employees with more than 20 years of experience in distribution of
    # both employees with relevant experience and no relevant experience

    looking_for_job = train_dataset_local[train_dataset_local['target'] == 1]
    not_looking = train_dataset_local[train_dataset_local['target'] == 0]

    # visualize enrolled in university
    pie_plot(looking_for_job, 'enrolled_university', 'Looking for jobs')
    pie_plot(not_looking, 'enrolled_university', 'Not looking for jobs')
    # more employees with full time course are looking for jobs
    # more employees without enrolment in university are NOT looking for a job

    # visualize education level
    pie_plot(looking_for_job, 'education_level', 'Looking for jobs')
    pie_plot(not_looking, 'education_level', 'Not looking for jobs')
    # similar results

    # visualize company size
    bar_plot(train_dataset_local, 'company_size', hue='target')

    # visualize for last new job
    bar_plot(train_dataset_local, 'last_new_job', hue='target')


def encoding_features(x):
    rel_exp_idx, edu_idx, comp_size_idx = [list(x.columns).index(col) for col in
                                           ['relevent_experience', 'education_level', 'company_size']]
    x.iloc[:, rel_exp_idx] = x.iloc[:, rel_exp_idx].map({'No relevent experience': -1,
                          'Has relevent experience': 1}).astype(int)
    x.iloc[:, edu_idx] = x.iloc[:, edu_idx].map({'Primary School': -1,
                                                        'High School': 1,
                                                        'Graduate': 2,
                                                        'Masters': 3,
                                                        'Phd': 4}).astype(int)
    x.iloc[:, comp_size_idx] = x.iloc[:, comp_size_idx].map({'<10': -1,
                                                             '10/49': 1,
                                                             '50-99': 2,
                                                             '100-500': 3,
                                                             '500-999': 4,
                                                             '1000-4999': 5,
                                                             '5000-9999': 6,
                                                             '10000+': 7}).astype(int)
    x.loc[(x['experience'] == '>20'), 'experience'] = 25
    x.loc[(x['experience'] == '<1'), 'experience'] = -1
    x.loc[(x['last_new_job'] == 'never'), 'last_new_job'] = -1
    x.loc[(x['last_new_job'] == '>4'), 'last_new_job'] = 5
    return x


###########################################
#                                         #
#            Visualising the data         #
#                                         #
###########################################


train_dataset = pd.read_csv('aug_train.csv')
# visualize_data(train_dataset_local=train_dataset)

# target: 0 – Not looking for job change, 1 – Looking for a job change

########################################
#                                      #
#       Cleaning up the data           #
#                                      #
########################################

print(train_dataset.info())
print()
print(train_dataset.describe(include='all'))
print()
# it shows that enrolled_university, education_level, experience and last_new_job features
# have less than 3% missing values, so will drop them
print(train_dataset.isnull().mean() * 100)
print()
train_dataset.dropna(subset=['enrolled_university', 'education_level', 'experience', 'last_new_job'], inplace=True)
print(train_dataset.isnull().mean() * 100)
print()


# While looking at the data, I could see that the "enrollee_id" is a unique number assigned for each employee so
# it is not a feature. Dropping it
train_dataset.drop('enrollee_id', axis=1, inplace=True)
print(train_dataset.head())
# city and city_development_index are having similar information, they are strictly linked together so dropping city
train_dataset.drop('city', axis=1, inplace=True)

X = train_dataset.drop('target', axis=1)
y = train_dataset['target']
# replacing missing values
# these are the supported strategies: allowed_strategies = ["mean", "median", "most_frequent", "constant"]
mode_imputer = SimpleImputer(strategy="most_frequent")
mode_imputer.fit(X)
X_imputed = mode_imputer.transform(X)
print(X_imputed)
X_df = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

# Label encoding relevant_experience, education_level, company_size, experience and last_new_job features
encoder = FunctionTransformer(encoding_features)
X_encoded = encoder.fit_transform(X_df)
print(X_encoded)

X_encoded['experience'] = X_encoded['experience'].astype(int)
X_encoded['last_new_job'] = X_encoded['last_new_job'].astype(int)
X_encoded['city_development_index'] = X_encoded['city_development_index'].astype(float)
X_encoded['training_hours'] = X_encoded['training_hours'].astype(int)
print(X_encoded)
print(X_encoded.info())


# encoding and feature scaling remaining stuff
num_attribs = ['city_development_index', 'training_hours']
cat_attribs = ['gender', 'enrolled_university', 'major_discipline', 'company_type']
preprocessing = ColumnTransformer([
    ("num", StandardScaler(), num_attribs),
    ("cat", OneHotEncoder(drop='first', sparse=False), cat_attribs)
], remainder="passthrough")

hr_processed = preprocessing.fit_transform(X_encoded)
print(hr_processed)

X_train, X_test, y_train, y_test = train_test_split(hr_processed, y, test_size=0.2, shuffle=True, stratify=y)
print(X_train)
print(y_test)
print(X_test)
print(y_test)


classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# test_dataset = pd.read_csv('aug_test.csv')
# X_test = test_dataset.iloc[:, :-1].values
# y_test = test_dataset.iloc[:, -1].values
#
# # Look at the data
