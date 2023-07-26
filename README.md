# credit-risk-classification
Module 20 Assignment - Supervised Learning

- Use of various techniques to train and evaluate a model based on loan risk. We use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
## Overview of the Analysis

### Split the Data into Training and Testing Sets
* Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
* Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining 
  columns.
* Split the data into training and testing datasets by using train_test_split.  

### Note: A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.  

### Create a Logistic Regression Model with the Original Data
* Fit a logistic regression model by using the training data (X_train and y_train).
* Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.
* Evaluate the model’s performance by doing the following:
  * Generate a confusion matrix.
  * Print the classification report.
  * Calculate the accuracy score of the model
    ![image](https://github.com/BijoyetaK/credit-risk-classification/assets/126313924/7093ff32-3d87-4c95-9ea7-cd61fc36cd56)

* Question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?
* Answer:
  The logistic regression model was 95% accurate at predicting the healthy vs high-risk loan labels. Precision: Out of all 
  the loans that the model predicted would be high-risk loan, 85% turned out to be so. Recall: Out of all the loans that 
  actually turned out to be high-risk, the model predicted this outcome correctly for 91% of those loans. F1 Score: This 
  value is calculated as: F1 Score: 2 * (Precision * Recall) / (Precision + Recall) F1 Score: 2 * (.85 * .91) / (.85 + .91) 
  F1 Score: 0.8789. F1 Score is very closer to 1 and fair score. This tells us that the model does a good job of predicting 
  whether or not loans pose as a high-risk loan.


### Predict a Logistic Regression Model with Resampled Training Data
* Use the RandomOverSampler module from the imbalanced-learn library to resample the data.Making sure to confirm that the 
  labels have an equal number of data points.

  ![image](https://github.com/BijoyetaK/credit-risk-classification/assets/126313924/0b9b3668-5e03-4f71-b67b-837f0a0d8c80)

### Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.
 ![image](https://github.com/BijoyetaK/credit-risk-classification/assets/126313924/4e10898b-18ee-4d0b-b7be-72bd11c0b4ac)

###  Evaluate the model’s performance by doing the following:
* Generate a confusion matrix.
* Print the classification report.
* Calculate the accuracy score of the model.

  ![image](https://github.com/BijoyetaK/credit-risk-classification/assets/126313924/a385e6c8-f705-4338-a38d-e8a59c2dbf2c)


#### Question : How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan)  and  1 (high-risk loan) labels?

#### Answer: The logistic regression model predicts the oversampled data of loans with near-perfect accuracy (99% accurate) of healthy loan vs high-risk loans.

### references: 
* Module activities
* https://www.geeksforgeeks.org/compute-classification-report-and-confusion-matrix-in-python/
* https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd
* https://www.statology.org/sklearn-classification-report/
* https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
* https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
* https://muthu.co/understanding-the-classification-report-in-sklearn/



