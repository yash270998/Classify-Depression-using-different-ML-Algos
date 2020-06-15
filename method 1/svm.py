import pandas as pd

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn import svm

col_names = ['X','Y','Class']
features = ['X','Y']
# load dataset
df = pd.read_csv("preprocesseddata.csv",usecols=col_names)
print(df.head())

x = df[features]
y = df.Class

# x.fillna(x.mean(), inplace=True)
# Target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10) 
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy using SVM model with this datamodel is :",metrics.accuracy_score(y_test, y_pred))
