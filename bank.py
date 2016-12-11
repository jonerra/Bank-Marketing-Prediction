import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


# import data from excel to a data frame
df = pd.read_csv('bank-full.csv', delimiter=';')

# drop columns contact and poutcome as they are missing too much data to add value
df = df.drop(['contact', 'poutcome'], 1)

# replace unknown values with NaN values for easier use later
df.replace('unknown', np.nan, inplace=True)

# fill missing values under Job and Eduction with the mode
df['job'] = df['job'].fillna(df['job'].mode()[0])
df['education'] = df['education'].fillna(df['education'].mode()[0])

# Print out first 5 rows of our altered data frame
print(df.head())

# Values must be assigned a number value to be used by machine learning algorithms
# LabelEncoder assigns variables to numbers for us
lbl = preprocessing.LabelEncoder()

# Converting job values to integers
lbl.fit(np.unique(list(df['job'].values)))
df['job'] = lbl.transform(list(df['job'].values))

# Converting marital values to integers
lbl.fit(np.unique(list(df['marital'].values)))
df['marital'] = lbl.transform(list(df['marital'].values))

# Converting education values to integers
lbl.fit(np.unique(list(df['education'].values)))
df['education'] = lbl.transform(list(df['education'].values))

# We can also map values to an integer value using a dictionary instead of LabelEncoder
# Mapping default values to integers
df['default'] = df['default'].map({'no': 0, 'yes': 1, 'unknown': 2})

# Mapping housing values to integers
df['housing'] = df['housing'].map({'no': 0, 'yes': 1, 'unknown': 2})

# Mapping loan values to integers
df['loan'] = df['loan'].map({'no': 0, 'yes': 1, 'unknown': 2})

# Mapping contact values to integers
# df['contact'] = df['contact'].map({'telephone': 0, 'cellular': 1, 'unknown': 2})

# Converting month values to integers
lbl.fit(np.unique(list(df['month'].values)))
df['month'] = lbl.transform(list(df['month'].values))

# Converting poutcome values to integers
# df['poutcome'] = df['poutcome'].map({'failure': 0, 'success': 1, 'other': 2, 'unknown': 3})

# Convert our data frame to multidimensional arrays
# The X variable is an array of the independent variables and drops column 'y'
X = np.array(df.drop(['y'], 1))

# The y variable is an array of the dependent variable 'y'
y = np.array(df['y'])

# Shuffle and partition our data into 80% train data and 20% test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Choosing which model to use for our data
clf = RandomForestClassifier(n_estimators=100)
# clf = LogisticRegression()
# clf = tree.DecisionTreeClassifier()
# clf = LinearRegression()

# Training the data
clf.fit(X_train, y_train)

# Scoring the model
accuracy = clf.score(X_test, y_test)

# Printing the score of the model
print(accuracy)

# Testing the predictions
# prediction = clf.predict(X)
# df['y_prediction'] = clf.predict(X)

# print(df.head())
