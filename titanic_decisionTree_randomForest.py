#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

#Import the Numpy library
import numpy as np
#Import 'tree' from scikit-learn library
from sklearn import tree
# Import the Pandas library
import pandas as pd
# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
# print("Print the `head` of the train and test dataframes")
# print(train.head())
# print(test.head())

# train.describe()
# test.describe()

# train.shape
# test.shape

# Passengers that survived vs passengers that passed away
print("Passengers that survived vs passengers that passed away")
print(train["Survived"].value_counts())

# As proportions
print("Passengers that survived vs passengers that passed away as proportions")
print(train["Survived"].value_counts(normalize = True))

# Males that survived vs males that passed away
print("Males that survived vs males that passed away")
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print("Females that survived vs females that passed away")
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print("Normalized male survival")
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

# Normalized female survival
print("Normalized female survival")
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train.loc[train["Age"] < 18, "Child"] = 1
train.loc[train["Age"] >= 18, "Child" ] = 0
# print("Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.")
# print(train["Child"])

# Print normalized Survival Rates for passengers under 18
print("Print normalized Survival Rates for passengers under 18")
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print("Print normalized Survival Rates for passengers 18 or older")
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

#Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

#Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
print("imputed embarked train")

train["Age"] = train["Age"].fillna(train["Age"].median())

#Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

#Print the Sex and Embarked columns
# print("Print the Sex and Embarked columns")
# print(train["Sex"])
# print(train["Embarked"])

# Print the train data to see the available features
print("train")
print(train)

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
print("features one")
print(features_one)

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
print("initialized my tree one")
my_tree_one = my_tree_one.fit(features_one, target)
print("my tree one")
print(my_tree_one)

# Look at the importance and score of the included features
print("Look at the importance and score of the included features from my tree one")
print("importance features from my tree one")
print(my_tree_one.feature_importances_)
print("score of the included features from my tree one")
print(my_tree_one.score(features_one, target))

#Convert the male and female groups to integer form
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

#Impute the Embarked variable
test["Embarked"] = test["Embarked"].fillna("S")
print("imputed embarked for test")

test["Age"] = test["Age"].fillna(test["Age"].median())

#Convert the Embarked classes to integer form
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2
# Impute the missing value with the median
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
print("test features")
print(test_features)

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)
# print("my prediction using test features")
# print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print("my solution")
print(my_solution)

# Check that your data frame has 418 entries
print("my solution shape, check it has 418 entries")
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
print("features two")
print(features_two)

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)
print("my tree two")
print(my_tree_two)

#Print the score of the new decison tree
print("Print the score of my tree two")
print("my tree two feature_importances_")
print(my_tree_two.feature_importances_)
print("my tree two target")
print(my_tree_two.score(features_two, target))

# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)
print("my tree three")
print(my_tree_three)

# Print the score of this decision tree
print("Print the score of my tree three")
print(my_tree_three.score(features_three, target))

#### Random forest ####
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
print("features forest")
print(features_forest)

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)
print("my forest")
print(my_forest)

# Print the score of the fitted random forest
print("Print the score of the fitted random forest")
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
# print("my prediction forest")
# print(pred_forest)
print("length of prediction vector")
print(len(pred_forest))

#Request and print the `.feature_importances_` attribute
print("print the `.feature_importances_` attribute")
print("my tree two")
print(my_tree_two.feature_importances_)
print("my forest")
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print("Compute and print the mean accuracy score for both models")
print("my tree two")
print(my_tree_two.score(features_two, target))
print("my forest")
print(my_forest.score(features_two, target))

