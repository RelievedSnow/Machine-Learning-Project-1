# importing dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collections And Pre-processing

# loading a dataset to a pandas dataframe

sonar_data = pd.read_csv('C:/Users/DELL/PycharmProjects/MLprojects/Copy of sonar data.csv', header=None)
# print(sonar_data.head())

# No. of rows and columns
# print(sonar_data.shape)
# print(sonar_data.describe())  # 25%,50%,75% percentile means that amt of values are less than the given value.

# to count how many rocks and mines are there

# print(sonar_data[60].value_counts())

# finding mean to group the data according to mine and rock
# print(sonar_data.groupby(60).mean())

# separating numerical data and labels

X = sonar_data.drop(60, axis=1)
Y = sonar_data[60]
# print(X)
# print(Y)

# Training and Test Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# print(X.shape)
# print(X_train.shape)
# print(X_test.shape)

# Model Training--> Logistic Regression

model = LogisticRegression()

# Training the logistic regression model with training data

# print(X_train, Y_train)
model.fit(X_train, Y_train)

# Model Evaluation--> accuracy of the model
# accuracy of the training model
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#print('Training Data Accuracy:', train_data_accuracy)

# accuracy of the test model
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

#print('Test Data Accuracy:', test_data_accuracy)

# Making a Prediction System

input_data = (0.0217,0.0152,0.0346,0.0346,0.0484,0.0526,0.0773,0.0862,0.1451,0.2110,0.2343,0.2087,0.1645,0.1689,0.1650,0.1967,0.2934,0.3709,0.4309,0.4161,0.5116,0.6501,0.7717,0.8491,0.9104,0.8912,0.8189,0.6779,0.5368,0.5207,0.5651,0.5749,0.5250,0.4255,0.3330,0.2331,0.1451,0.1648,0.2694,0.3730,0.4467,0.4133,0.3743,0.3021,0.2069,0.1790,0.1689,0.1341,0.0769,0.0222,0.0205,0.0123,0.0067,0.0011,0.0026,0.0049,0.0029,0.0022,0.0022,0.0032)  # empty initialization

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# as the model takes 2-D array as input we are providing it 1-D array to overcome that we reshape it into 2-D array

input_array_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_array_reshaped)
# print(prediction)

if prediction[0] == 'R':  # 0 represents the first element of the list
    print('The Object is a Rock')
else:
    print('The object is a Mine')