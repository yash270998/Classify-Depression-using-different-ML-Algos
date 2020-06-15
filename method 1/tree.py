
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from time import time
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree

from sklearn import svm
TS_LENGTH = 2000
TRAINSPLIT = 652 / 752
RANDOMSTATE = np.random.randint(1, 2**16)

root = os.path.curdir

condition = np.load(
    os.path.join(root, "condition_{}_emb.npy".format(TS_LENGTH)))
control = np.load(os.path.join(root, "control_{}_emb.npy".format(TS_LENGTH)))

X = np.concatenate((condition, control), axis=0)
y = to_categorical(np.array([0] * len(condition) + [1] * len(control)))
print(X,y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - TRAINSPLIT, random_state=RANDOMSTATE)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy using Decision Tree Classifier is :",metrics.accuracy_score(y_test, y_pred))
    