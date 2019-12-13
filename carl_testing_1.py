
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC, ADASYN
from collections import Counter
from keras.utils import to_categorical

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}


train_transactions = pd.read_csv("./data/train_transaction.csv")
train_transactions.isFraud.value_counts()
filtered_transactions = train_transactions.iloc[:,0:17]

print(filtered_transactions.shape)
print(train_transactions.shape)
filtered_transactions.head(100)
encoded_dataset = pd.get_dummies(filtered_transactions, columns=['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain'], drop_first=True)

encoded_dataset.fillna(0, inplace=True)

y = encoded_dataset.isFraud.values

X = encoded_dataset.drop(columns='isFraud').values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

class_weights = get_class_weights(y)
class_weights = {0:1, 1:28}

rfclassifier = RandomForestClassifier(verbose=2, class_weight=class_weights, n_jobs=7, n_estimators = 10)
rfclassifier.fit(X_train,y_train)

y_pred  = rfclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))