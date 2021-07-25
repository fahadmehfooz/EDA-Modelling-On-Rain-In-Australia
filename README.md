# EDA-Modelling-On-Rain-In-Australia: Project Overview 
* Predicting next day rain by training classification models on the target variable RainTomorrow.
* Dataset contains 145460 rows and 23 columns.
* Dealt with class imbalance using SMOTE.
* Trained on multiple models like Logistic Regression, XGBoost, Bernoulli Naive Bayes, Gaussian Naive Bayes, and RandomForest.
## Languages Used 
**Python Version:** 3.9.0

## Resources and Tools Used
**Tools:** Jupyter Notebook

**Packages:** Pandas, NumPy, sklearn, Matplotlib, missingno, seaborn, imblearn and counter

## Data Used
**Data Source**: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

## Data Wrangling and Data Visualization
* Null values found, filling missing values using appropriate functions.
* Converting categorical columns to continuous.
* Plotting heatmap to find the correlation between continuous features.
* Plotting violinplots, barplots and boxplots for the features.
* Removing the outliers.
* Oversampling the train data to get more balanced data.

## Model Building 

I took a split on the data with training data as 80% and test data as 20%. 
I tried different models like Logistic Regression, XGBoost, Bernoulli Naive Bayes, Gaussian Naive Bayes, and RandomForest.

## Metrics Used For Evaluation

* Accuracy
* F1-Score

## Model performance

**Results:**

* The accuracy of the Logistic Regression model is:  76.66697387217178 %
  F1 score for the logistic regression model is: 76.46361586910545 %

* The accuracy of XGBoost model is:  90.68245702962997 %
  F1 score for the XGBoost model is: 90.41751575754702 %

* The accuracy of the Gaussian Naive Bayes model is:  74.42974978111607 %
  F1 score for the Gaussian Naive Bayes model is: 73.42178369575629 %

* The accuracy of Bernoulli Naive Bayes model is :  72.21326206165615 %
  F1 score for Bernoulli Naive Bayes is : 72.04709809011682 %

* The accuracy of the Random Forest model is:  90.66402469932261 %
