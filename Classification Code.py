#Importing all the libraries that we will need

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


#Importing the data and setting the index to datetime format

dataset = pd.read_csv("Book4-2.csv")
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.set_index('Date')



#Creating the additional differential columns

data = dataset.iloc[1826:, 0:]
data['Interest Differential'] = data['Turkey Interest Rate'] - data['US Interest Rate']
data['Turkey Public Debt '] = data['Turkey Public Debt ']/1000
data['Turkey Public Debt'] = data['Turkey Public Debt ']
data['Growth Rate Differential'] = data['Turkey GDP Growth Rate'] - data['US GDP Growth Rate']


#Picking out only the relevant columns (in order these were the Current Account Balance, Inflation RD, Unemployment Diff, Interest RD, External Public Debt, GDP Growth RD) 
X = data.iloc[:, [5, 6, 10, 13, 14, 15]].values
y = data.iloc[:, [0]].values


#Function to take averages of the data columns over a specified period

def chunk_it_up(array, n):
    
    size = np.shape(array)
    rows = size[0]
    columns = size[1]
    
    largest = int(rows/n)
    remainder = rows%n
    
    avg_matrix = np.zeros([largest, columns])
    
    for i in range(largest):
        for j in range(columns):
             avg_matrix[i][j] = np.mean(array[(i*n):(i*n+n), j])
             
    rem_matrix = []
    
    if remainder > 0:
        for j in range(columns):
            rem_matrix.append(np.mean(array[(largest*n):, j]))
            
        rem_matrix = np.reshape(rem_matrix, [1, len(rem_matrix)])
    
        avg_matrix = np.append(avg_matrix, rem_matrix, axis = 0)
        
    
    return avg_matrix
    

X_5 = chunk_it_up(X, 5)
y_5 = chunk_it_up(y, 5)



#Function to make the (-1, 0, 1) labels

def convert(array):
    size = np.shape(array)
    rows = size[0]
    columns = size[1]
    
    converted_array = np.zeros([rows-1, columns])
    
    for i in range(1, rows):
        for j in range(columns):
            if array[i][j] > array[i-1][j]:
                converted_array[i-1][j] = -1
            elif array[i][j] < array[i-1][j]:
                converted_array[i-1][j] = 1
    
    return converted_array


X_5 = convert(X_5)
y_5 = convert(y_5)



#Creating training and test sets for the weekly accuracy analysis

X_train_1 = X_5[0:-1, :]
X_test_1 = X_5[-1:, :]
y_train_1 = y_5[0:-1, :]
y_test_1 = y_5[-1:, :]

X_train_2 = X_5[0:-2, :]
X_test_2 = X_5[-2:, :]
y_train_2 = y_5[0:-2, :]
y_test_2 = y_5[-2:, :]

X_train_4 = X_5[0:-4, :]
X_test_4 = X_5[-4:, :]
y_train_4 = y_5[0:-4, :]
y_test_4 = y_5[-4:, :]

X_train_12 = X_5[0:-12, :]
X_test_12 = X_5[-12:, :]
y_train_12 = y_5[0:-12, :]
y_test_12 = y_5[-12:, :]


#Function to conduct the weekly accuracy analysis and collate the results in a dictionary

def lets_evaluate(classifier):
    classifier.fit(X_train_1, y_train_1)
    pred_for_1 = classifier.predict(X_test_1)
    cm_1 = confusion_matrix(y_test_1, pred_for_1)
    
    classifier.fit(X_train_2, y_train_2)
    pred_for_2 = classifier.predict(X_test_2)
    cm_2 = confusion_matrix(y_test_2, pred_for_2)
    
    classifier.fit(X_train_4, y_train_4)
    pred_for_4 = classifier.predict(X_test_4)
    cm_4 = confusion_matrix(y_test_4, pred_for_4)
    
    classifier.fit(X_train_12, y_train_12)
    pred_for_12 = classifier.predict(X_test_12)
    cm_12 = confusion_matrix(y_test_12, pred_for_12)
    
    
    classifier.fit(X_5, y_5)
    pred_full = classifier.predict(X_5)
    
    acc_full = confusion_matrix(pred_full, y_5)

    output_dict = {
            'Confusion Matrix for last week': cm_1,
            'Confusion Matrix for last 2 weeks': cm_2,
            'Confusion Matrix for last 4 weeks': cm_4,
            'Confusion Matrix for last 10 weeks': cm_12,
            'Overall accuracy': acc_full
            }
    
    return output_dict


#Creating and performing tests with several classifier models

classifier_rbf = SVC(kernel = 'rbf', gamma = 'auto')
matrix_rbf = lets_evaluate(classifier_rbf)

classifier_poly_2 = SVC(kernel = 'poly', degree = 2, gamma = 'auto')
matrix_poly_2 = lets_evaluate(classifier_poly_2)

classifier_poly_3 = SVC(kernel = 'poly', degree = 3, gamma = 'auto')
matrix_poly_3 = lets_evaluate(classifier_poly_3)

classifier_linear = SVC(kernel = 'linear', gamma = 'auto')
matrix_linear = lets_evaluate(classifier_linear)

classifier_dt = DecisionTreeClassifier(criterion = 'entropy')
matrix_dt = lets_evaluate(classifier_dt)

classifier_dt_gini = DecisionTreeClassifier(criterion = 'gini')
matrix_dt_gini = lets_evaluate(classifier_dt_gini)


#K-fold cross validation for robustness checks

avg_rbf = sum(cross_val_score(classifier_rbf, X_7, y_7, cv=83))/83
avg_dt = sum(cross_val_score(classifier_dt, X_7, y_7, cv=83))/83
avg_poly_2 = sum(cross_val_score(classifier_poly_2, X_7, y_7, cv=83))/83
avg_poly_3 = sum(cross_val_score(classifier_poly_3, X_7, y_7, cv=83))/83
avg_linear = sum(cross_val_score(classifier_linear, X_7, y_7, cv=83))/83
avg_gini = sum(cross_val_score(classifier_dt_gini, X_7, y_7, cv = 83))/83
