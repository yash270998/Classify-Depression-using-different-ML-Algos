import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['afftype','Variance','Depressed']
# load dataset
pima = pd.read_csv("scoresdeviation.csv", header=None, names=col_names)
pima.head()
feature_cols = ['afftype','Variance']

X = pima[feature_cols] # Features
X = X.apply(pd.to_numeric, errors='coerce')
y = pima.Depressed
X.fillna(X.mean(), inplace=True)
 # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
