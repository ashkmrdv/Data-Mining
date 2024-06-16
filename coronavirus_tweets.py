# Part 3: Text mining.

import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import urllib.request
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	dataFrame = pd.read_csv(data_file,encoding='latin-1')
	
	return dataFrame

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	sen=list(df['Sentiment'].unique())

	return sen

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	count = (df.groupby(['Sentiment']).size()).nlargest(2)
	Value=str(count.idxmin())

	return(Value)

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	Dataframe=df.loc[df['Sentiment'] == 'Extremely Positive']

	count = (Dataframe.groupby(['TweetAt']).size()).nlargest(2)
	Value=str(count.idxmax())
	
	return Value

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	Arr=np.array(df['OriginalTweet'])

	for i in range(len(Arr)):
		Arr[i]=Arr[i].lower()

	df['OriginalTweet']=pd.Series(Arr)

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	Arr=np.array(df['OriginalTweet'])

	for i in range(len(Arr)):

		for j in range(len(Arr[i])):
			if (Arr[i][j].isalpha()):
				continue
			else: 
				Arr[i]=Arr[i].replace(Arr[i][j],' ')
			

	df['OriginalTweet']=pd.Series(Arr)

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	Arr=np.array(df['OriginalTweet'])

	for i in range(len(Arr)):
		res = Arr[i].split()
		for j in range (len(res)):
			res[j]=res[j].strip()
		Arr[i]=" ".join(res)		
		
	df['OriginalTweet']=pd.Series(Arr)

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	Arr=np.array(df['OriginalTweet'])
	for i in range(len(Arr)):
		Arr[i]=Arr[i].split()
	
	df['OriginalTweet']=pd.Series(Arr)
	
# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	Arr=np.array(tdf['OriginalTweet'])
	total=0
	for i in range(len(Arr)):
		total=total+len(Arr[i])

	return total

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	Arr=np.array(tdf['OriginalTweet'])
	total=0
	Arr_2=[]
	for i in range(len(Arr)):
		for j in range(len(Arr[i])):
			Arr_2.append(Arr[i][j])

	unique_values = list(set(Arr_2))
	total=len(unique_values)

	return total

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	Arr=np.array(tdf['OriginalTweet'])
	List=[]
	freq=[]
	for i in range(len(Arr)):
		for j in range(len(Arr[i])):
			List.append(Arr[i][j])
	counts = Counter(List)
	total=counts.most_common(k)
	for i in range(len(total)):
		freq.append(total[i][0])		

	return list(freq)

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	link = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"

	f = urllib.request.urlopen(link)
	myfile = list(f.read())
	for i in range (len(myfile)):
		myfile[i]=chr(myfile[i])
	myfile=''.join(myfile)
	myfile=myfile.split()

	Arr=np.array(tdf['OriginalTweet'])
	Arr_2=[]

	for i in range(len(Arr)):
		List=Arr[i]
		Final_Arr=[]
		for j in range(len(List)):
			if (((len(List[j]))>2)&(not(List[j] in myfile))):
				Final_Arr.append(List[j])			
		Arr_2.append(Final_Arr)		
	tdf['OriginalTweet']=pd.Series(Arr_2)	

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	Arr=np.array(tdf['OriginalTweet'])
	Arr_2=[]
	ps = PorterStemmer()
	for i in range(len(Arr)):
		List=Arr[i]
		Final_Arr=[]
		for j in range(len(List)):
			Final_Arr.append(ps.stem(List[j]))			
		Arr_2.append(Final_Arr)		
	tdf['OriginalTweet']=pd.Series(Arr_2)


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):

	df_2 = df[['OriginalTweet', 'Sentiment']]
	df_2= df_2.rename(columns={'OriginalTweet': 'text', 'Sentiment': 'label'})

	X = df_2['text']
	y = df_2['label']

	vectorizer = CountVectorizer(max_df=20,min_df=1,max_features=80000)
	X_vec = vectorizer.fit_transform(X)
	mnb = MultinomialNB(alpha=0.8, fit_prior=True, force_alpha=True)
	mnb.fit(X_vec, y)
	y_pred = np.array(mnb.predict(X_vec))

	return (y_pred)
		
# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	acc = accuracy_score(y_pred, y_true)
	acc=round(acc,3)

	return acc

# Main 
# The following is the Test program that has been used. Please un comment it if needs to be tested against the inputs that I used.


# PATH="C:/Users/ashkm/OneDrive/Desktop/DATA MINING ASSESSMENT/data_sets/data/coronavirus_tweets.csv"

# DF=read_csv_3(PATH)
# DF_2=read_csv_3(PATH)

# SENT=get_sentiments(DF)
# print("The list of Sentiments")
# print(SENT)

# SECSEN=second_most_popular_sentiment(DF)
# print("The second most popular Sentiment")
# print(SECSEN)

# MOST=date_most_popular_tweets(DF)
# print("The Date with the most Extremely positive tweets")
# print(MOST)

# lower_case(DF)
# print("The Lower cased messages")
# print(DF)

# remove_non_alphabetic_chars(DF)
# print("Non Aplhabetic characters removed messages")
# print(DF)

# remove_multiple_consecutive_whitespaces(DF)
# print("Remove consecutive white Spaces")
# print(DF)

# tokenize(DF)
# print("List of words")
# print(DF)

# TOT=count_words_with_repetitions(DF)
# print("Total number of words")
# print(TOT)

# TOT_DIS=count_words_without_repetitions(DF)
# print("Total number of distinct words")
# print(TOT_DIS)

# FREQ=frequent_words(DF,10)
# print("Top 10 frequent words")
# print(FREQ)

# remove_stop_words(DF)
# print("Stop words removed")
# print(DF)

# stemming(DF)
# print("Stemming words")
# print(DF)

# MNB=mnb_predict(DF_2)
# print("Predicted MNB values")
# print(MNB)

# LABEL=np.array(DF_2['Sentiment'])
# ACC=mnb_accuracy(MNB,LABEL)
# print("Accuracy score")
# print(ACC)
  
