#-------------------------------------------------------------------------
# AUTHOR(S): Amir Soleymani, Sarah Liu
# FILENAME: decision_tree.py
# SPECIFICATION: CART decision tree for age prediction
# FOR: CS 4210- Final Project
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('NHANES_age_prediction.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

for i in range(len(db)):
   X.append([])
   
   #add the features to the vector X.
   for j in range(10):
      if j >= 3:
         X[i].append(float(db[i][j]))

      #add the age group label to the vector Y.
      elif j == 1:
         Y.append(db[i][j])

# split the dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)

#fitting the depth-4 decision tree to the data using entropy as your impurity measure
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)

# train the model using only the training data
clf.fit(X_train, Y_train)

# use the trained model to predict the test data
Y_pred = clf.predict(X_test)

# print evaluation scores
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred, pos_label="Senior"))
print("Recall:", recall_score(Y_test, Y_pred, pos_label="Senior"))
print("F1 Score:", f1_score(Y_test, Y_pred, pos_label="Senior"))

# using sklearn built in metrics to get more info
# u can ignore this
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

#plotting decision tree
tree.plot_tree(clf, feature_names=['Gender', 'Physical', 'BMI', 'Fasting Glucose', 'Diabetic', 'Oral', 'Insulin'], class_names=['Adult','Senior'], filled=True, rounded=True, fontsize=5)
plt.show()