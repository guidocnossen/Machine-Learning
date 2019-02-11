from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.cluster import KMeans
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, homogeneity_completeness_v_measure, adjusted_rand_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import cross_val_score, StratifiedKFold

import sys
import time

# preprocessing function that divided the input data into two list
# documents list for all the reviews in the file

def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file) as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0])
                
    return list(zip(documents,labels))
 
# a dummy function that just returns its input
def identity(x):
    return x

def main(argv):
	if len(sys.argv) != 3:
		print("Usage: 'python' 'LFDassignment3_SVM_group7.py' '<trainset>' '<testset>'")
	else:
		# define x(data) and y(label) by handling the input files
		# Note: I used 75% of the trainset.text as training data, 0.125% for the testset and 0.125% for the devset 
		print('reading data...')
		train = read_corpus(sys.argv[1],use_sentiment=True)
		test = read_corpus(sys.argv[2],use_sentiment=True)
		split_point = int(0.50*len(test))
		test = test[split_point:]
		dev = test[:split_point]
		# define Xtrain, Xtest, Ytrain, Ytest, Xdev, Ydev
		Xtrain = [i[0] for i in train]
		Xtest = [i[0] for i in test]
		Ytrain = [i[1] for i in train]
		Ytest = [i[1] for i in test]
		Xdev = [i[0] for i in dev]
		Ydev = [i[1] for i in dev]
		

		tfidf = True
		# TdifdVectorizer with additional features used for classification
		# I used only stopwords
		if tfidf:
			vec = TfidfVectorizer(preprocessor = identity,
								  tokenizer = identity,
								  stop_words = 'english', 
								  ngram_range=(1,2),
								  sublinear_tf=True, 
								  use_idf=True)
		else:
			vec = CountVectorizer(preprocessor = identity,
								  tokenizer = identity)

		# define the Support Vector Model with a linear kernel
		clf = svm.SVC(kernel='linear', C=0.95)
		
		# define the Kmeans classifier with 6 clusters
		'''clf = KMeans(n_clusters=6, n_init=2, verbose=1)'''
		classifier = Pipeline([('vec', vec), ('cls', clf)])
		
		print('training data...')
		# train the classifier with features and their labels
		classifier.fit(Xtrain,Ytrain)
		
		print('predicting values...')
		# predict values of Xtest and Xdev
		Yguess = classifier.predict(Xtest)
		Yguess2 = classifier.predict(Xdev)
		print()
		# calculate the accuracy scores for Test set and Dev set for the SVM classifier 
		accuracy = accuracy_score(Ytest, Yguess)
		accuracy2 = accuracy_score(Ydev, Yguess2)
		average = ((accuracy + accuracy2) / 2)
		
		print(('accuracy on test set:', accuracy))
		print('-'*40)
		
		print(('accuracy on dev set:', accuracy2))
		print('-'*40)
		
		print(('average system accuracy:', average))
		
		# calculate accuracy for the Kmeans classifier
		'''try:
			print(classifier.labels_)
		except:
			pass
		print(adjusted_rand_score(Ytest,Yguess))
		print(homogeneity_completeness_v_measure(Ytest,Yguess))'''
		
		
		# calculate the precision, recall, f-score and confusion matrix
		'''print('Classification report on Test set:')
		print(classification_report(Ytest, Yguess))
		
		print('Classification report on Test set:')
		print(classification_report(Ydev, Yguess2))

		# use bottom function for confusion matrix
		print('Confusion matrix for Test set:')
		print(confusion_matrix(Ytest, Yguess))
		
		print('Confusion matrix for Dev set:')
		print(confusion_matrix(Ydev, Yguess2))'''

if __name__ == "__main__":
    main(sys.argv)
