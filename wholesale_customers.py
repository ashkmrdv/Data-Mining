# Part 2: Cluster Analysis

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

OMP_NUM_THREADS=2

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	dataFrame = pd.read_csv(data_file,encoding='unicode_escape')
	dataFrame=dataFrame.drop(columns=["Channel","Region"])

	return dataFrame

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	DF=df.describe(include='all')
	summary=(((DF.drop('count')).drop('25%')).drop('50%')).drop('75%')
	summary=summary.round(0)
	summary=summary.astype(int)

	return summary
	
# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):

	scaler = StandardScaler()
	df_new = scaler.fit_transform(df)
	df_new=pd.DataFrame(df_new,columns=df.columns.values)
	
	return(df_new)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
	kmeans = KMeans(k)
	kmeans.fit(df)
	y=pd.Series(kmeans.labels_)
	
	return y	

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	pass

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	clustering = AgglomerativeClustering(n_clusters=k).fit(df)
	y=pd.Series(clustering.labels_)
	
	return(y)
	

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
	silhouette = silhouette_score(X,y)

	return silhouette

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	arr_k=np.zeros((30,4))
	arr_k_std=np.zeros((30,4))
	arr_a=np.zeros((30,4))
	arr_a_std=np.zeros((30,4))

	for i in range (30):
		arr_k[i][0]=0
		arr_k[i][1]=0
		
		if (i%3==0):
			Y=kmeans(df,3)
			c=clustering_score(df,Y)
			arr_k[i][2]=3
			arr_k[i][3]=c
			
		elif (i%3==1):
			Y=kmeans(df,5)
			c=clustering_score(df,Y)
			arr_k[i][2]=5
			arr_k[i][3]=c
			
		elif(i%3==2):
			Y=kmeans(df,10)
			c=clustering_score(df,Y)
			arr_k[i][2]=10
			arr_k[i][3]=c
			
	for i in range (30):
		arr_a[i][0]=1
		arr_a[i][1]=0

		if (i%3==0):
			Y=agglomerative(df,3)
			c=clustering_score(df,Y)
			arr_a[i][2]=3
			arr_a[i][3]=c
			
		elif (i%3==1):
			Y=agglomerative(df,5)
			c=clustering_score(df,Y)
			arr_a[i][2]=5
			arr_a[i][3]=c
			
		elif(i%3==2):
			Y=agglomerative(df,10)
			c=clustering_score(df,Y)
			arr_a[i][2]=10
			arr_a[i][3]=c
			
	df_std=standardize(df)
	
	for i in range (30):
		arr_k_std[i][0]=0
		arr_k_std[i][1]=1
		
		if (i%3==0):
			Y_std=kmeans(df_std,3)
			c_std=clustering_score(df_std,Y_std)
			arr_k_std[i][2]=3
			arr_k_std[i][3]=c_std
			
		elif (i%3==1):
			Y_std=kmeans(df_std,5)
			c_std=clustering_score(df_std,Y_std)
			arr_k_std[i][2]=5
			arr_k_std[i][3]=c_std

		elif(i%3==2):
			Y_std=kmeans(df_std,10)
			c_std=clustering_score(df_std,Y_std)
			arr_k_std[i][2]=10
			arr_k_std[i][3]=c_std
	
	for i in range (30):
		arr_a_std[i][0]=1
		arr_a_std[i][1]=1
		
		if (i%3==0):
			Y_std=agglomerative(df_std,3)
			c_std=clustering_score(df_std,Y_std)
			arr_a_std[i][2]=3
			arr_a_std[i][3]=c_std
			
		elif (i%3==1):
			Y_std=agglomerative(df_std,5)
			c_std=clustering_score(df_std,Y_std)
			arr_a_std[i][2]=5
			arr_a_std[i][3]=c_std

		elif(i%3==2):
			Y_std=agglomerative(df_std,10)
			c_std=clustering_score(df_std,Y_std)
			arr_a_std[i][2]=10
			arr_a_std[i][3]=c_std
	
	Arr=np.concatenate((arr_k,arr_k_std,arr_a,arr_a_std))
	
	DataFrame=pd.DataFrame(Arr,columns=['Algorithm','data','k','Silhouette Score'])
	DataFrame['Algorithm'].replace([0.0,1.0],['K Means', 'Agglomerative'], inplace=True)
	DataFrame['data'].replace([0.0,1.0],['Original', 'Standardized'], inplace=True)
	
	return DataFrame
	
# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	best = rdf['Silhouette Score'].max()

	return(best)

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	df_std=standardize(df)
	kmeans = KMeans(n_clusters=3)
	kmeans.fit(df_std)
	cols=list(df.columns.values)

	for i in range (len(cols)):
		j=i+1
		while(j<len(cols)):
			plt.scatter(df_std[cols[i]],df_std[cols[j]],c=kmeans.labels_)
			plt.xlabel( cols[i] , fontsize=12 )
			plt.ylabel( cols[j] , fontsize=12 )
			plt.title( cols[i]+" Vs "+cols[j], fontsize=12 )
			plt.savefig('../TEMPLATE_FILES/' + cols[i]+' Vs '+cols[j] +'.pdf' )
			j=j+1

# The following is the Test program that has been used. Please un comment it if needs to be tested against the inputs that I used.

# PATH="C:/Users/ashkm/OneDrive/Desktop/DATA MINING ASSESSMENT/data_sets/data/wholesale_customers.csv"

# DF=read_csv_2(PATH)
# print(DF)

# SUMMARY=summary_statistics(DF)
# print("Summary statistics")
# print(SUMMARY)

# STANDARD=standardize(DF)
# print("Standardized Data")
# print(STANDARD)

# KMEANS=kmeans(DF,3)
# print("K Means Clustering Data")
# print(KMEANS)

# AGGL=agglomerative(DF,3)
# print("Agglomerative Clustering")
# print(AGGL)

# SILH_1=silhouette_score(DF,KMEANS)
# SILH_2=silhouette_score(DF,AGGL)
# print("Silhouette score of K means")
# print(SILH_1)
# print("Silhouette score of Agglomerative")
# print(SILH_2)

# CE=cluster_evaluation(DF)
# print("Data Frame of Cluster Evaluation")
# print(CE)

# BEST=best_clustering_score(CE)
# print("Best Score")
# print(BEST)

# PLOT=scatter_plots(DF)
