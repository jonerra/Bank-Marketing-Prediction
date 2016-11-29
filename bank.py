import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

df = pd.read_csv('bank-full.csv', delimiter=';')
df = df.drop(['contact', 'poutcome'], 1)
df.replace('unknown', np.nan, inplace=True)
df['job'] = df['job'].fillna(df['job'].mode()[0])
df['education'] = df['education'].fillna(df['education'].mode()[0])
print(df.head())

lbl = preprocessing.LabelEncoder()

# job
lbl.fit(np.unique(list(df['job'].values)))
df['job'] = lbl.transform(list(df['job'].values))

# marital
lbl.fit(np.unique(list(df['marital'].values)))
df['marital'] = lbl.transform(list(df['marital'].values))

# education
lbl.fit(np.unique(list(df['education'].values)))
df['education'] = lbl.transform(list(df['education'].values))

# default
df['default'] = df['default'].map({'no': 0, 'yes': 1, 'unknown': 2})

# housing
df['housing'] = df['housing'].map({'no': 0, 'yes': 1, 'unknown': 2})

# loan
df['loan'] = df['loan'].map({'no': 0, 'yes': 1, 'unknown': 2})

# contact
# df['contact'] = df['contact'].map({'telephone': 0, 'cellular': 1, 'unknown': 2})

# month
lbl.fit(np.unique(list(df['month'].values)))
df['month'] = lbl.transform(list(df['month'].values))

# poutcome
# df['poutcome'] = df['poutcome'].map({'failure': 0, 'success': 1, 'other': 2, 'unknown': 3})

X = np.array(df.drop(['y'], 1))
y = np.array(df['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

# prediction = clf.predict(X)
# df['y_prediction'] = clf.predict(X)

# print(df.head())
