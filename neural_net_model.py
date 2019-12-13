import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping

from keras.models import Sequential
import tensorflow as tf


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}

train_transactions = pd.read_csv("./data/train_transaction.csv")

filtered_transactions = train_transactions.iloc[:,0:17]

encoded_dataset = pd.get_dummies(train_transactions, columns=['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain'], drop_first=True)
encoded_dataset.fillna(0, inplace=True)

y = encoded_dataset.isFraud.values

X = encoded_dataset.drop(columns='isFraud').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


model = Sequential()
model.add(Dense(100, input_dmim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
print(model.summary())

es = [EarlyStopping(monitor='acc', mode='auto', verbose=1, patience=2, restore_best_weights=True)]

history = model.fit(X_train,
                 y_train,
                 epochs=3,
                 batch_size=64,
                 verbose=1,
                 class_weight=class_weights,
                 callbacks = es,
                 validation_split=0.2,
                 )

y_pred  = model.predict_classes(X_test)

Counter(y_pred[:,0])
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
