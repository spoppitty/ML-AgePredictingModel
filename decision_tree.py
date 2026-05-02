#-------------------------------------------------------------------------
# AUTHOR(S): Amir Soleymani
# FILENAME: decision_tree.py
# SPECIFICATION: CART decision tree for age prediction
# FOR: CS 4210- Final Project
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
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
         X[i].append(db[i][j])

#add the age group label to the vector Y.
      elif j == 1:
         Y.append(db[i][j])

#fitting the depth-4 decision tree to the data using entropy as your impurity measure
#--> add your Python code here
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4,)
clf.fit(X, Y)

print(X[0])
#plotting decision tree
tree.plot_tree(clf, feature_names=['Gender', 'Physical', 'BMI', 'Fasting Glucose', 'Diabetic', 'Oral', 'Insulin'], class_names=['Adult','Senior'], filled=True, rounded=True, fontsize=5)
plt.show()