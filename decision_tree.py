#-------------------------------------------------------------------------
# AUTHOR(S): Amir Soleymani, Sarah Liu
# FILENAME: decision_tree.py
# SPECIFICATION: CART decision tree for age prediction
# FOR: CS 4210- Final Project
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
import matplotlib.pyplot as plt
import csv

FEATURE_NAMES = ['Gender', 'Physical', 'BMI', 'Fasting Glucose', 'Diabetic', 'Oral', 'Insulin']
POSITIVE_LABEL = "Senior"

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

def print_scores(model_name, true_labels, predicted_labels):
   print("\n" + model_name)
   print("Accuracy:", accuracy_score(true_labels, predicted_labels))
   print("Precision:", precision_score(true_labels, predicted_labels, pos_label=POSITIVE_LABEL, zero_division=0))
   print("Recall:", recall_score(true_labels, predicted_labels, pos_label=POSITIVE_LABEL))
   print("F1 Score:", f1_score(true_labels, predicted_labels, pos_label=POSITIVE_LABEL))
   print("\nClassification Report:")
   print(classification_report(true_labels, predicted_labels, zero_division=0))

# keep the original depth-4 tree as a baseline for comparison
baseline_clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
baseline_clf.fit(X_train, Y_train)
baseline_pred = baseline_clf.predict(X_test)
print_scores("Baseline depth-4 decision tree", Y_test, baseline_pred)

# tune the tree for Senior-class F1 instead of general accuracy
focused_senior_weights = [{'Adult': 1, 'Senior': weight} for weight in [3, 4, 5, 6]]
focused_param_grid = {
   'criterion': ['entropy', 'gini'],
   'max_depth': [4, 5, 6],
   'min_samples_split': [2, 20, 40],
   'min_samples_leaf': [5, 10, 20],
   'class_weight': ['balanced'] + focused_senior_weights
}

broad_senior_weights = [{'Adult': 1, 'Senior': weight} for weight in [2, 3, 4, 5, 6, 7, 8, 10]]
broad_param_distributions = {
   'criterion': ['entropy', 'gini'],
   'splitter': ['best', 'random'],
   'max_depth': [3, 4, 5, 6, 7, 8, 10, None],
   'min_samples_split': [2, 5, 10, 20, 40, 60, 80],
   'min_samples_leaf': [1, 2, 5, 10, 15, 20, 30],
   'max_leaf_nodes': [None, 8, 12, 16, 24, 32, 48],
   'min_impurity_decrease': [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01],
   'class_weight': ['balanced'] + broad_senior_weights
}

f1_scorer = make_scorer(f1_score, pos_label=POSITIVE_LABEL)
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

focused_search = GridSearchCV(
   tree.DecisionTreeClassifier(random_state=42),
   focused_param_grid,
   scoring=f1_scorer,
   cv=cross_validation,
   n_jobs=1
)

# randomized search checks many more parameter types without trying every possible combination
broad_search = RandomizedSearchCV(
   tree.DecisionTreeClassifier(random_state=42),
   broad_param_distributions,
   n_iter=300,
   scoring=f1_scorer,
   cv=cross_validation,
   n_jobs=1,
   random_state=42
)

focused_search.fit(X_train, Y_train)
broad_search.fit(X_train, Y_train)

searches = [
   ("Focused grid search", focused_search),
   ("Broad randomized search", broad_search)
]
best_search_name, best_search = max(searches, key=lambda search: search[1].best_score_)
clf = best_search.best_estimator_

# use the trained model to predict the test data
Y_pred = clf.predict(X_test)

print("\nSearch results:")
for search_name, search_model in searches:
   print(search_name + " best cross-validation F1:", search_model.best_score_)

print("\nSelected search:", best_search_name)
print("Best parameters:")
print(best_search.best_params_)
print("Best cross-validation F1:", best_search.best_score_)
print_scores("Tuned decision tree", Y_test, Y_pred)

#plotting decision tree
plt.figure(figsize=(18, 10))
tree.plot_tree(clf, feature_names=FEATURE_NAMES, class_names=clf.classes_, filled=True, rounded=True, fontsize=7)
plt.tight_layout()
plt.show()
