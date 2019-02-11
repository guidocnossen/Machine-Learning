import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from nltk.stem import SnowballStemmer, PorterStemmer
from sklearn.metrics import accuracy_score, f1_score, classification_report, homogeneity_completeness_v_measure, adjusted_rand_score
import re, string, unicodedata
import nltk
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction import stop_words

import sys

np.random.seed(1337)


def read_corpus(text_file, task):

	text = [] # create a list for the text
	labels = [] # create a list for the text-labels
	ids = [] # create a list for the ids
	with open(text_file, encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split()
			ids.append(parts[0]) # append the first part of the line (id) to the id list
			text.append(parts[3:]) # append all the elements of the list from the 3 part of the line to the text list
			# define different classification tasks
			# each classification task has its own labels or label combinations
			if task == 'hyperp':
				labels.append(parts[1])
			if task == 'bias':
				labels.append(parts[2])
			if task == 'joint-assignment':
				if parts[1] == 'True' and parts[2] == 'left':
					labels.append('True/Left')
				if parts[1] == 'True' and parts[2] == 'right':
					labels.append('True/Right')
				if parts[1] == 'False' and parts[2] == 'left-center':
					labels.append('False/Left-Center')
				if parts[1] == 'False' and parts[2] == 'right-center':
					labels.append('False/Right-Center')
				if parts[1] == 'False' and parts[2] == 'least':
					labels.append('False/Least')
					
	return ids, text, labels
	
def preprocessor(words):

	# preprocessor
	stemmer = SnowballStemmer('english')
	new_words = []
	for word in words:
		new_word = re.sub(r'[^\w\s]', '', word)
		new_word = stemmer.stem(word)
		if new_word != '':
			new_words.append(new_word)
			
	return new_words
	
def identity(x):
	return x
		
def main(argv):
	
	# define a list of tasks -- the same as in the read corpus function -- to iterate over
	tasks = ['hyperp', 'bias', 'joint-assignment']
	for i in tasks:
		# define train and test sets and their corresponding sets of ID's
		IDtrain, Xtrain, Ytrain = read_corpus(argv[1], i)
		IDtest, Xtest, Ytest = read_corpus(argv[2], i)

		# define vectorizer + classifier and combine them in a pipeline
		pipeline = Pipeline([('vec', TfidfVectorizer(tokenizer = identity, preprocessor = identity)),
							 ('clf', LinearSVC(class_weight='balanced'))])
		
		# fit the training data to the model -- train the model
		print('training model...')
		model = pipeline.fit(Xtrain, Ytrain)
		
		# predict the values for the test data
		print('predicting...')
		y_pred = model.predict(Xtest)
		
		# calculate accuracy scores
		# calculate precision, recall, f1 scores
		# calculate f1 macro scores
		print('-'*80)
		print('Scores for {0} classification:'.format(i))
		print(accuracy_score(y_pred, Ytest))
		print(classification_report(Ytest, y_pred))
		print(f1_score(Ytest, y_pred, average='macro'))

		# create an output file with both hyperp labels and bias labels in combination with their ID's
		# use the joint-assignment for this task -- this task contains both hyperp and bias label information
		if i == 'joint-assignment':
			# create an output_dic_true for the hyperp True labels 
			# create an output_dic_false for the hyperp False labels 
			output_dic_true = {}
			output_dic_false = {}
			
			# create an output_dic to combine the dictionaries from above
			output_dic = {}
			
			# combine the ID's with the predicted values
			op = list(zip(IDtest,y_pred))
			
			# iterate over these combined lists and append desired data to dictionaries
			for i in op:
				if 'True' in i[1]:
					dic = {i[0] : [i[1][0:4], i[1][5:]] }
					output_dic_true.update(dic)
				else:
					dic = { i[0] : [i[1][0:5], i[1][6:]] }
					output_dic_false.update(dic)
			# combine seperate dictionaries into one dictionary
			output_dic_true.update(output_dic_false)
			output_dic.update(output_dic_true)
			
			# define a title column for the output file
			output = 'ID, Hyperp-value, Bias-value\n'
			
			# write the data from the dictionaries to the outputfile alongside the names in the title rule
			# each ID and their corresponding prediction values are written to a new row
			with open('output.csv', 'w') as f:
				f.write(output)
				for key in output_dic.keys():
					ids = key
					hyperp = output_dic[key][0]
					bias = output_dic[key][1]
					row = ids + " , " + hyperp + " , " + bias + "\n"
					f.write(row)
			f.close()				
if __name__=="__main__":
	main(sys.argv)
