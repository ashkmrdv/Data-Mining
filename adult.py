# Part 1: Decision Trees with Categorical Attributes

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error as mae 

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
	dataFrame = pd.read_csv(data_file,encoding='unicode_escape')
	dataFrame.drop(columns=['fnlwgt'],inplace=True)
	
	return dataFrame

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	rows=df.index.size
	return rows

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	cols=list(df.columns.values)
	return cols

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	missingValues=df.isnull().sum().sum()
	return missingValues
	
# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	df_2=df.isnull().sum()
	array=df_2.to_numpy().nonzero()[0]
	cols=np.array(df.columns.values)
	columnsWithMissingValues=[]
	for i in range (len(array)):
		columnsWithMissingValues.append(cols[array[i]])
	columnsWithMissingValues=list(columnsWithMissingValues)

	return columnsWithMissingValues

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	bachelor=df['education'].value_counts()['Bachelors']
	master=df['education'].value_counts()['Masters']
	rows=df.index.size
	percentage=round((bachelor+master)/rows*100,1)

	return percentage

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	rows=df.index.size
	df_null=df.isnull()
	df_new=[]

	for i in range (rows):
		if ((any(df_null.iloc[i]))==False):
			df_new.append(df.iloc[i])

	df_new_2 = pd.DataFrame(df_new)

	return df_new_2

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	df_without_missing=data_frame_without_missing_values(df)
	cols=df_without_missing.columns
	num_cols=df_without_missing._get_numeric_data().columns
	categorical_attributes=sorted(list(set(cols)-set(num_cols)))
	categorical_attributes.remove("class")

	one_hot_encoded_data = (pd.get_dummies(df_without_missing, columns = categorical_attributes)).drop(columns=["class"])
	
	return one_hot_encoded_data

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	df_without_missing=data_frame_without_missing_values(df)
	cols=list(df_without_missing['class'])
	label_encoder = preprocessing.LabelEncoder()
	label_encoded_series=pd.Series(label_encoder.fit_transform(cols))

	return label_encoded_series

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	dtree = DecisionTreeClassifier()
	dtree = dtree.fit(X_train, y_train)
	return(pd.Series(dtree.predict(X_train)))

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	error = mae(y_true, y_pred) 
	return(error)

# The following is the Test program that has been used. Please un comment it if needs to be tested against the inputs that I used.

# PATH="C:/Users/ashkm/OneDrive/Desktop/DATA MINING ASSESSMENT/data_sets/data/adult.csv"
# DF=read_csv_1(PATH)
# print("The Data Frame Read")
# print(DF)

# ROWS=num_rows(DF)
# print("Number of Instances")
# print(ROWS)

# COLS=column_names(DF)
# print("Number of Attributes")
# print(COLS)

# MV=missing_values(DF)
# print("Total Number of Missing Values")
# print(MV)

# CMV=columns_with_missing_values(DF)
# print("Columns with Missing Values")
# print(CMV)

# BM=bachelors_masters_percentage(DF)
# print("Percentage of People with Bachelor's or Master's degree")
# print(BM)

# DF=data_frame_without_missing_values(DF)
# print("Data Frame without Missing Values")
# print(DF)

# ONEHOT=one_hot_encoding(DF)
# print("One hot encoded Data Frame")
# print(ONEHOT)

# LABEL=label_encoding(DF)
# print("Labels of Instances after Label encoding")
# print(LABEL)

# TRAIN_X = ONEHOT
# TRAIN_Y = LABEL
# PREDICT_Y=dt_predict(TRAIN_X,TRAIN_Y)
# print("The predicted Value")
# print(PREDICT_Y)

# ERROR_RATE=dt_error_rate(PREDICT_Y,TRAIN_Y)
# print("Error Rate of the Classifier")
# print(ERROR_RATE)