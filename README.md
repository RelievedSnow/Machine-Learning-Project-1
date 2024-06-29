# Machine-Learning-Projects
Project 1 : SONAR Rock vs Mine Prediction with Python
Dataset Link: https://drive.google.com/file/d/1pQxtljlNVh0DHYg-Ye7dtpDTlFceHVfa/view

# Step 1: Fisrt we importing dependencies
-Such as numpy, pandas, train_test_split from sklearn.model_selection, LogisticRegression from sklearn.linear_model and at last accuracy_score from sklearn.metrics.S
-We use numpy Library for storing values as array, Pandas Library for reading the dataset from .csv file and train_test_split( to split the dataset int training and test data --We use Logistic Regression Model for prediction(rock or mine). 
-We use accuracy_score to check how accurately our model predictes from trained data and test data.

# Step 2: Data Pre-Processing and Data Collestion.
-We use the 'pd' variable to capture the data 
-We use the '.shape' method to check the size of the dataset
-We use the '.value_counts' function on the labelled data to count the total no. of 'Mine' and 'Rocks'.
-As the values of Mine and Rock are somewhat close we use the Mean Function to find the mean value of 'Rock' and 'Mine' in each feature.
-Now we seperate the Numerical Data from the Labelled Data and Store the in 'X' and 'Y' variables.

# Step 3: Training and Testing Data
- We store the training data in 'X_train', and corresponding training label data in 'Y_train'.
- We store the test data in 'X_test' and corresponding test label in 'Y_test'.

# Step 4: Now put the data for training purpose into the Logistic Regression Model.
-We use the model.fit method the input the 'X_train' and 'Y_train' data into the Model.
-We use the '.predict' function to perform prediction.
-We check the accuracy of the trained data using the accuraccy_score function.
-Similarly we perform the same steps on the test data.

# Step 5: Now we Make the Prediction.
-We input the value of each Feature int he 'input_data' variable.
-We reshape(1, -1) the input value into a single row (first dimension) and infers the number of columns (second dimension) based on your original data.
-Once you've reshaped your data appropriately, you can use the 2D array for prediction.
