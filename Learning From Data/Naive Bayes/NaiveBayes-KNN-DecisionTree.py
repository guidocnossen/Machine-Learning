from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn import tree
import time
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# load in the corpus/data, get the labels(Y) for each task (sentiment, topic)
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
                labels.append( tokens[0] )

    return documents, labels

# a dummy function that just returns its input
def identity(x):
    return x
def main():
    print()
    if len(sys.argv) != 3:
        print("Usage: 'pythonversion' 'pythonfile' 'trainset' 'testset'")
    else:
        # define x(data) and y(label), split the data in test and train
        Xtrain, Ytrain = read_corpus(sys.argv[1], use_sentiment=False)
        Xtest, Ytest = read_corpus(sys.argv[2], use_sentiment=False)

        # let's use the TF-IDF vectorizer
        tfidf = True

        # we use a dummy function as tokenizer and preprocessor,
        # since the texts are already preprocessed and tokenized.
        if tfidf:
            vec = TfidfVectorizer(preprocessor = identity,
        	                  tokenizer = identity,
                              stop_words = 'english')
        else:
            vec = CountVectorizer(preprocessor = identity,
        	                  tokenizer = identity)

        
        mnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        dtc = tree.DecisionTreeClassifier()
        knc = KNeighborsClassifier(n_neighbors=9)
        neigh = NearestNeighbors()
        clflist = [mnb,dtc,knc]
        resdict = {}
        for i in clflist:
            print("running:",i)
            classifier = Pipeline( [('vec', vec),
            	                ('cls', i)] )

            # fit the traindata(Xtrain) and the trainlabels(Ytrain) for the classifier
            t0 = time.time()
            classifier.fit(Xtrain, Ytrain)
            train_time = time.time() - t0
            print("training time: ", train_time)

            # get the predicted y's(labels)
            t0 = time.time()
            Yguess = classifier.predict(Xtest)
            test_time = time.time() - t0
            print("test time: ", test_time)

            # print results, (mostly) comparing the predicted y's(Yguess labels) to the wright y's(Ytest labels)
            print("acc:", accuracy_score(Ytest, Yguess))
            print("f1:", f1_score(Ytest, Yguess, average='macro', labels=classifier.classes))
            print()
            if resdict != {}:
                for k,v in resdict.items():
                    if accuracy_score(Ytest, Yguess) > v[1]:
                        resdict.clear()
                        resdict[i] = ["acc:", accuracy_score(Ytest, Yguess),"prec:", precision_score(Ytest, Yguess, average='macro', labels=classifier.classes_),
                        "rec:", recall_score(Ytest, Yguess, average='macro', labels=classifier.classes_),"f1:", f1_score(Ytest, Yguess, average='macro', labels=classifier.classes_),
                        "cm:", classifier.classes,confusion_matrix(Ytest, Yguess,labels=classifier.classes_),"prob:", classifier.classes_, classifier.predict_proba(Xtest)]
            else:
                resdict[i] = ["acc:", accuracy_score(Ytest, Yguess),"prec:", precision_score(Ytest, Yguess, average='macro', labels=classifier.classes_),
                "rec:", recall_score(Ytest, Yguess, average='macro', labels=classifier.classes_),"f1:", f1_score(Ytest, Yguess, average='macro', labels=classifier.classes_),
                "cm:", classifier.classes_,confusion_matrix(Ytest, Yguess,labels=classifier.classes_),"prob:", classifier.classes_, classifier.predict_proba(Xtest)]

        for k,v in resdict.items():
            print("best performance:",k)
            for n in v:
                print(n)
        print()
if __name__ == "__main__":
    main()
