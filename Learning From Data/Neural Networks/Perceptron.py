import numpy, json, argparse
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import itertools

numpy.random.seed(1337)

# Read in the NE data, with either 2 or 6 classes
def read_corpus(corpus_file, binary_classes):
	print('Reading in data from {0}...'.format(corpus_file))
	words = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split()
			words.append(parts[0])
			if binary_classes:
				if parts[1] in ['GPE', 'LOC']:
					labels.append('LOCATION')
				else:
					labels.append('NON-LOCATION')
			else:
				labels.append(parts[1])	
	print('Done!')
	return words, labels

# Read in word embeddings 
def read_embeddings(embeddings_file):
	print('Reading in embeddings from {0}...'.format(embeddings_file))
	embeddings = json.load(open(embeddings_file, 'r'))
	embeddings = {word:numpy.array(embeddings[word]) for word in embeddings}
	print('Done!')
	return embeddings

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(words, embeddings):
	vectorized_words = []
	for word in words:
		try:
			vectorized_words.append(embeddings[word.lower()])
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
    
 
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='KerasNN parameters')
	parser.add_argument('data', metavar='named_entity_data.txt', type=str, help='File containing named entity data.')
	parser.add_argument('embeddings', metavar='embeddings.json', type=str, help='File containing json-embeddings.')
	parser.add_argument('-b', '--binary', action='store_true', help='Use binary classes.')
	args = parser.parse_args()
	
	# Read in the data and embeddings
	X, Y = read_corpus(args.data, binary_classes = args.binary)
	embeddings = read_embeddings(args.embeddings)
	
	# Transform words to embeddings
	X = vectorizer(X, embeddings)
	print(X)
	
	# Transform string labels to one-hot encodings
	encoder = LabelBinarizer()
	Y = encoder.fit_transform(Y) # Use encoder.classes_ to find mapping of one-hot indices to string labels
	if args.binary:
		Y = numpy.where(Y == 1, [0,1], [1,0])
		
	# Split in training and test data
	split_point = int(0.75*len(X))
	Xtrain = X[:split_point]
	Ytrain = Y[:split_point]
	Xtest = X[split_point:]
	Ytest = Y[split_point:]
	
	print(X.shape, Y.shape)

	# Define properties of the baseline classifier - for both binary and 6-way classification
	if args.binary:
		baseline_clf = DummyClassifier(strategy='most_frequent',random_state=0)
	else:
		baseline_clf = DummyClassifier()
	
	# Define the properties of the perceptron model
	model = Sequential()
	model.add(Dense(input_dim = X.shape[1], units = Y.shape[1]))
	model.add(Activation("linear"))
	sgd = SGD(lr = 0.01)
	loss_function = 'mean_squared_error'
	model.compile(loss = loss_function, optimizer = sgd, metrics=['accuracy'])
	
	# Train the perceptron model and the baseline classifier
	model.fit(Xtrain, Ytrain, verbose = 1, epochs = 1, batch_size = 32)
	baseline_clf.fit(Xtrain, Ytrain)
	
	# Get predictions of the perceptron model and the baseline classifier
	Yguess = model.predict(Xtest)
	Yguess_dummy = baseline_clf.predict(Xtest)
	
	# Convert to numerical labels to get scores with sklearn in 6-way setting
	Yguess = numpy.argmax(Yguess, axis = 1)
	Ytest = numpy.argmax(Ytest, axis = 1)
	Yguess_dummy = numpy.argmax(Yguess_dummy, axis = 1)
		
	# use this rule for to see how three random word are classified by using the pre-trained word embeddings as input
	# I used the words Rotterdam, Adidas and Chroesjtsjov
	'''print(encoder.classes_[numpy.argmax(model.predict(vectorizer(['Rotterdam', 'Adidas', 'Chroesjtsjov'], embeddings)), axis = 1)])'''
	
	# Accuracy scores for the baseline classifier and the perceptron model
	print('Classification accuracy on test - Baseline Classifier: {0}'.format(accuracy_score(Ytest, Yguess_dummy)))
	print()
	print('Classification accuracy on test - Perceptron Model: {0}'.format(accuracy_score(Ytest, Yguess)))
	
	# Error analysis
	# Recall, Precision and F1 scores for all the classes
	print()
	print('Classification report - Baseline Classifier:')
	print(classification_report(Ytest, Yguess_dummy))
	
	print('Classification report - Perceptron Model:')
	print(classification_report(Ytest, Yguess))
	
	#Confusion matrices
	print('Plotting Confusion Matrices...:')
	cm = confusion_matrix(Ytest, Yguess_dummy)
	cm2 = confusion_matrix(Ytest, Yguess)
	np.set_printoptions(precision=2)
	print('Done!')
	# plot confusion matrices with visualization function
	plt.figure()
	plot_confusion_matrix(cm, classes=encoder.classes_,
					  title='Confusion matrix - Baseline Classifier')
	
	plt.figure()
	plot_confusion_matrix(cm2, classes=encoder.classes_,
					  title='Confusion matrix - Perceptron Model')
	plt.show()
