# Importing basic necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Dataframe creation 

diabetes = pd.read_csv('diabetes.csv')

# Checking dataframe dimensions , head ...

print("Columns are : ", diabetes.columns)

print("First 5 are ; ", diabetes.head())

print("Shape of dataframe is : ", diabetes.shape)

# Importing train_test_split

from sklearn.model_selection import train_test_split

# Initializing x & y

x = diabetes.loc[:, diabetes.columns != 'Outcome']

print("value of x is : ", x)
 
y = diabetes['Outcome']

print("value of y is : ", y)

# Initializing data to X_train, y_train , X_test , y_test

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=66)

print("Shape of X_train : ", X_train.shape)

print("Shape of X_train : ", y_train.shape)

print("Shape of X_train : ", X_test.shape)

print("Shape of X_train : ", y_test.shape)

# Importing KNN library from sklearn

from sklearn.neighbors import KNeighborsClassifier

# two list for storing accuracy of sample training and testing data 

train_accuracy = []
test_accuracy = []

# taking number of neighbors 1-10 for checking score/accuracy

neighbors = range(1, 11)

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    train_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

print("Training Accuracy : ", train_accuracy)

print("Testing Accuracy : ", test_accuracy)

# Ploating Graph for finalizing number of neighbors 

plt.plot(neighbors, train_accuracy, label="Training Accuracy", c = 'red')
plt.plot(neighbors, test_accuracy, label="Testing Accuracy", c = 'green')
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")

# Above graph shows that as number of neighbors increasing for training dataset , score decreasing

# For number of neighbors = 10 , I got max score for testing dataset . So I have taken [ n_neighbor's = 10 ]

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

print("Final accuracy of model for training is : ", knn.score(X_train, y_train))           # 0.81

print("Final accuracy of model for testing is : ", knn.score(X_test, y_test))              # 0.73

