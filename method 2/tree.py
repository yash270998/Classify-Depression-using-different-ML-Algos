import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from time import time
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import random
from sklearn import svm
TS_LENGTH = 2000
TRAINSPLIT = 652 / 752
RANDOMSTATE = np.random.randint(1, 2**16)

ConditionGroupFileNames = os.listdir('data/condition')
ControlGroupFileNames = os.listdir('data/control')
X = []
y = []
for fileName in ConditionGroupFileNames:
    df = pd.read_csv('data/condition/'+str(fileName))
    dates = df['date'].unique()
    activityLevelsPerDay = []
    for date in dates:
        if len(df[df['date']==date]) == 1440:
            temp = pd.DataFrame(df[df['date']==date]).drop(columns=['timestamp','date'])
            activityLevelsPerDay.append(temp)
    for dailyActivityLevel in activityLevelsPerDay:
        activityVector = np.array(dailyActivityLevel["activity"])
        if len(activityVector) == 1440:
            X.append(activityVector)
            y.append(1)
for fileName in ControlGroupFileNames:
    df = pd.read_csv('data/control/'+str(fileName))
    dates = df['date'].unique()
    activityLevelsPerDay = []
    for date in dates:
        if len(df[df['date']==date]) == 1440:
            temp = pd.DataFrame(df[df['date']==date]).drop(columns=['timestamp','date'])
            activityLevelsPerDay.append(temp)
    for dailyActivityLevel in activityLevelsPerDay:
        activityVector = np.array(dailyActivityLevel["activity"])
        if len(activityVector) == 1440:
            X.append(activityVector)
            y.append(0)
    
combinedDict = list(zip(X, y))
random.shuffle(combinedDict)
X[:], y[:] = zip(*combinedDict)

X = np.array(X)
y = np.array(y)


print(X,y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - TRAINSPLIT, random_state=RANDOMSTATE)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy using Decision Tree Classifier is :",metrics.accuracy_score(y_test, y_pred))
    