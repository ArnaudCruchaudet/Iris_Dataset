#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:50:27 2024

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA




#############################Classification Task##############################

iris = datasets.load_iris()
X = iris.data[:,:]  
y = iris.target
X2 = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
## Score gives us the percentage of correct prediction in the testing sample 
knn.score(X_test,y_test)


predicted = knn.predict(X_test)
print("Predictions from the classifier:")
print(predicted)
print("Target values:")
print(y_test)




##changing the testing sample size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)


##changing the number of k 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
## Score gives us the percentage of correct prediction in the testing sample 
knn.score(X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
## Score gives us the percentage of correct prediction in the testing sample 
knn.score(X_test,y_test)



##changing the distance measure: Manhattan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 3, metric='manhattan')
knn.fit(X_train,y_train)
## Score gives us the percentage of correct prediction in the testing sample 
knn.score(X_test,y_test)


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 5, metric='manhattan')
knn.fit(X_train,y_train)
## Score gives us the percentage of correct prediction in the testing sample 
knn.score(X_test,y_test)



X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 3, weights='distance', metric='euclidean')
knn.fit(X_train,y_train)
knn.score(X_test,y_test)


#Detailed report of performance
y_pred = knn.predict(X_test)
classes_names = ['setosa','versicolor','virginica']
cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
sns.heatmap(cm, annot=True, fmt='d')


print(classification_report(y_test, y_pred))
#precision(true positive/(true positive + false positive)
#Recall(true positive/(true positive + false negative))
#F1(harmonic mean of two previous)




##Method to define k (classification)
f1s = []
# Calculating f1 score for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    # using average='weighted' to calculate a weighted average for the 3 classes 
    f1s.append(f1_score(y_test, pred_i, average='weighted'))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), f1s, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('F1 Score K Value')
plt.xlabel('K Value')
plt.ylabel('F1 Score')



######################Regression Task#################################

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = housing.data[:,:]  
y = housing.target
X2 = StandardScaler().fit_transform(X)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsRegressor(n_neighbors = 3)
knn.fit(X_train,y_train)
#Give you the R2
knn.score(X_test,y_test)

#Obtain prediction
y_pred = knn.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)


print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')
##Method to define k (regression)
error = []
# Calculating MAE error for K values between 1 and 39
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    mae = mean_absolute_error(y_test, pred_i)
    error.append(mae)
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)

plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')
#In our case, optimal number seems to be 8 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsRegressor(n_neighbors = 8)
knn.fit(X_train,y_train)
#Give you the R2
knn.score(X_test,y_test)



X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)
knn = KNeighborsRegressor(n_neighbors = 8)
knn.fit(X_train,y_train)
#Give you the R2
knn.score(X_test,y_test)
error = []
# Calculating MAE error for K values between 1 and 39
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    mae = mean_absolute_error(y_test, pred_i)
    error.append(mae)
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
#we still select k=8 but we see how scaling influence k-NN performance => 11



X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)
knn = KNeighborsRegressor(n_neighbors = 11)
knn.fit(X_train,y_train)
#Give you the R2
knn.score(X_test,y_test)



#Combining PCA and KNN
pca = PCA().fit(X2)
X_pca=pca.transform(X2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
X_pca2=X_pca[:, 0:6]



X_train, X_test, y_train, y_test = train_test_split(X_pca2, y, test_size=0.2, random_state=0)
knn = KNeighborsRegressor(n_neighbors = 8)
knn.fit(X_train,y_train)
#Give you the R2
knn.score(X_test,y_test)


error = []
# Calculating MAE error for K values between 1 and 39
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    mae = mean_absolute_error(y_test, pred_i)
    error.append(mae)
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)


X_train, X_test, y_train, y_test = train_test_split(X_pca2, y, test_size=0.2, random_state=0)
knn = KNeighborsRegressor(n_neighbors = 10)
knn.fit(X_train,y_train)
#Give you the R2
knn.score(X_test,y_test)



#Obtain prediction
y_pred = knn.predict(X_test)

x_ax=range(4128)
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=1.5, color="red", label="predicted")
plt.legend()
plt.show()