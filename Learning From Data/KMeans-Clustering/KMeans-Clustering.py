from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import TweetTokenizer
import re, string, unicodedata

from sklearn.metrics import accuracy_score, confusion_matrix, homogeneity_completeness_v_measure, adjusted_rand_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import cross_val_score
import random
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans

import sys
import os
import time

def read_corpus(directory):
	documents = []
	labels = []
	for root, directories, files in os.walk(directory):
		if 'train' in root:
			for filename in files:
				filepath = os.path.join(root,filename)
				with open(filepath) as f:
					tokens = f.readline().split()
					documents.append(tokens)
					labels.append(root[9:])
		else:
			for filename in files:
				filepath = os.path.join(root,filename)
				with open(filepath) as f:
					tokens = f.readline().split()
					documents.append(tokens)
					labels.append(root[8:])


	return list(zip(documents, labels))

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

def tokenizer(words):
	tknzr = TweetTokenizer()
	text = tknzr.tokenize(words)

	return text

# a dummy function that just returns its input
def identity(x):
    return x

def main(argv):

	print("Usage: python LFDassignment3_KMextra_Group10.py <C50trainset> <C50testset>")

	print('Reading Data...')
	# define train and test set
	# shuffle data
	train = read_corpus(sys.argv[1])
	test = read_corpus(sys.argv[2])
	random.shuffle(train)
	random.shuffle(test)
	# only use a part of the test data
	split_point = int(0.10*len(test))
	test = test[:split_point]
	Xtrain = [i[0] for i in train]
	Xtest = [i[0] for i in test]
	Ytrain = [i[1] for i in train]
	Ytest = [i[1] for i in test]

	tfidf = True

	# TdifdVectorizer with additional features used for classification
	# I used only stopwords
	if tfidf:
		vec = TfidfVectorizer(ngram_range=(1,3), analyzer='word', preprocessor = preprocessor,
							  tokenizer = identity,
							  stop_words = 'english',
							  lowercase = True)
	else:
		vec = CountVectorizer(ngram_range=(1,3), analyzer='word', preprocessor = preprocessor,
							  tokenizer = identity)

	# define the Support Vector Model with a linear kernel
	'''clf = svm.SVC(kernel='linear', C=1)'''
	# define the Kmeans classifier with 50 cluster
	clf = KMeans(n_clusters=50, random_state=1000, n_init=1, verbose=0)
	classifier = Pipeline([('vec', vec), ('cls', clf)])

	print('Training Classifier...')
	# train the classifier with features and their labels
	classifier.fit(Xtrain,Ytrain)

	print('Predicting Test Values...')
	# predict values of Xtest
	Yguess = classifier.predict(Xtest)

	# calculate the accuracy scores for the SVM classifier
	'''accuracy = accuracy_score(Ytest, Yguess)
	print(('Accuracy:', accuracy))'''
	print('-'*40)

	# calculate accuracy for the Kmeans classifier
	try:
		print(classifier.labels_)
	except:
		pass
	print(adjusted_rand_score(Ytest,Yguess))
	print(homogeneity_completeness_v_measure(Ytest,Yguess))

if __name__=="__main__":
	main(sys.argv)
