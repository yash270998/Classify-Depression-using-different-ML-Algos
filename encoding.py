import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['age1','gender','95th','90th','Std Deviation','Depressed']
# load dataset
df = pd.read_csv("scores.csv",usecols=col_names)
# print(df.head())
# print(df.keys())
# df_onehot = pd.get_dummies(df,columns=['gender'],prefix=['gender'])
# print(df_onehot.head())

feature_cols = ['age1','gender','95th','90th']
# print(df_onehot)
X = df[feature_cols] # Features
X = X.apply(pd.to_numeric, errors='coerce')
y = df.Depressed
X.fillna(X.mean(), inplace=True)
 # Target variable
accarray = []
for i in range(1,1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=i ) 
    # print(X_train)
    # print(y_test)
# Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    accarray.append(metrics.accuracy_score(y_test, y_pred))
sum =0
for i in range(0,999):
    sum = sum + accarray[i]
print(sum/1000)
# Model Accuracy, how often is the classifier correct?
