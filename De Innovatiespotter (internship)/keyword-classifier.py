import numpy, json, argparse
#from keras.models import Sequential
#from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score


#numpy.random.seed(1337)

# Read in the trefwoorden data
def read_corpus(corpus_file):
	print('Reading in data from {0}...'.format(corpus_file))
	words = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			label_list = []
			words_list = []
			parts = line.strip().split()
			for i in parts:
					if '__label__' in i:
						label_list.append(i)
					else:
						words_list.append(i)
			labels.append(label_list)	
			words.append(words_list)
			
	print('Done!')
	
	return words,labels

# Read in word embeddings 
def read_embeddings(embeddings_file):
	print('Reading in embeddings from {0}...'.format(embeddings_file))
	embeddings = json.load(open(embeddings_file, 'r', encoding='latin-1'))
	embeddings = {word:numpy.array(embeddings[word]) for word in embeddings}
	print('Done!')
	return embeddings
	
def vectorizer(words, embeddings):
	vectorized_words = []
	for i in words:
		try:
			vectorized_words.append(embeddings[i.lower()])
		except KeyError:
			vectorized_words.append(embeddings['UNK'])
	return numpy.array(vectorized_words)

#function for visualizing the confusion matrices
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print()

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    		
# a dummy function that just returns its input
def identity(x):
    return x
	
	
	# --------------------NAIVE BAYES CLASSIFIER-------------------------------
def main(argv):

	if len(sys.argv) != 2:
		print("Usage: 'python' 'keyword-classifier.py' '<trainset>'")
	else:
		# read in classified trefwoorden file
		corpus_file = 'data/Topsectoren/' + argv[1]
		X, Y = read_corpus(corpus_file)
		
		# divide data in training and test data
		split_point = int(0.80*len(X))
		Xtrain = X[split_point:]
		Ytrain = Y[split_point:]
		Xtest = X[:split_point]
		Ytest = Y[:split_point]
		# results for the word2vec model that can be used for prediction
		# values in this list are random and derived from a random query in the word2vec model
		X_word2vec = ['olie']
	
		# let's use the TF-IDF vectorizer
		tfidf = True

		# we use a dummy function as tokenizer and preprocessor,
		# since the texts are already preprocessed and tokenized.
		if tfidf:
			vec = TfidfVectorizer(preprocessor = identity,
								  tokenizer = identity)
		else:
			vec = CountVectorizer(preprocessor = identity,
								  tokenizer = identity)

		# combine the vectorizer with a Naive Bayes classifier
		classifier = Pipeline( [('vec', vec),
								('cls', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))] )
		# fit model
		classifier.fit(Xtrain, Ytrain)

		# predict model 
		Yguess = classifier.predict(Xtest)
		'''Y_random_guess = classifier.predict(X_word2vec)
		print(Y_random_guess)'''

		# return accuracy score
		print(accuracy_score(Ytest, Yguess))
		
		
		#return classification report
		#print('Classification report:')
		#print(classification_report(Ytest, Yguess))


		# return confusion matrix 
		#print('Confusion matrix:')
		#print(confusion_matrix(Ytest, Yguess))
	
if __name__=='__main__':
	main(sys.argv)

# ---------------------SUPPORT VECTOR MACHINE -- Linear SVC -----------------------------

'''def main(argv):
	
	if len(sys.argv) != 3:
		print("Usage: 'python' 'keyword-classifier.py' '<trainset>' '<testset>'")
	else:
		# define x(data) and y(label) by handling the input files
		# Note: I used 75% of the trainset.text as training data and the rest for test data
		Xtrain, Ytrain = read_corpus('data/Topsectoren/' + sys.argv[1])
		Xtest, Ytest = read_corpus('data/Topsectoren/' + sys.argv[2])
		tfidf = True
		
		# TdifdVectorizer with additional features used for classification
		if tfidf:
			vec = TfidfVectorizer(preprocessor = identity,
								  tokenizer = identity,
								  ngram_range=(1,2))
		else:
			vec = CountVectorizer(preprocessor = identity,
								  tokenizer = identity,
								  ngram_range=(1,2))

		# define the Support Vector Model with a linear kernel
		clf = svm.LinearSVC(C=2)
		classifier = Pipeline([('vec', vec), ('cls', clf)])
		
		# train the classifier with features and their labels
		classifier.fit(Xtrain,Ytrain)
		
		# predict values of Xtest
		Yguess = classifier.predict(Xtest)
		
		# calculate the accuracy scores
		accuracy = accuracy_score(Ytest, Yguess)
		print(('accuracy:', accuracy))
		print('-'*40)
		
		#apply cross validation on the trainset
		#scores = cross_val_score(clf, <param1>, <param2>, cv = 2)
		#print(scores)
		
		
		# calculate the precision, recall, f-score and confusion matrix
		print('Classification report:')
		print(classification_report(Ytest, Yguess))

		# use bottom function for confusion matrix
		print('Confusion matrix:')
		print(confusion_matrix(Ytest, Yguess))

if __name__ == "__main__":
    main(sys.argv)

