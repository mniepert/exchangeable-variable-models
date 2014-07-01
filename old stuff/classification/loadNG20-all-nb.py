from operator import mul
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from collections import Counter
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_mldata, load_files
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

NG20Categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

NG20Combinations = list(combinations(NG20Categories, 20))
numCombinations = len(list(NG20Combinations))
print "Number of unique pairs: ",numCombinations

errorNB = zeros(numCombinations, dtype=float)

a = 0

for categories in NG20Combinations:

	print "Loading 20 newsgroups dataset for categories:"
	print categories

	data_train_raw = fetch_20newsgroups(subset='train',  remove=('headers', 'footers', 'quotes'), categories=categories)
	print "%d documents" % len(data_train_raw.filenames)
	print "%d categories" % len(data_train_raw.target_names)

	data_test_raw = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
	print "%d documents" % len(data_test_raw.filenames)
	print "%d categories" % len(data_test_raw.target_names)


	# split a training set and a test set
	y_train, y_test = data_train_raw.target, data_test_raw.target

	vectorizer = CountVectorizer(binary=True)
	X_train = vectorizer.fit_transform(data_train_raw.data)
	X_test = vectorizer.transform(data_test_raw.data)

	#ch2 = SelectKBest(chi2, 20000)
	#X_train = ch2.fit_transform(X_train, y_train)
	#X_test = ch2.transform(X_test)

	m,n = X_train.shape

	print m, " ", n

	clf = BernoulliNB(alpha=0.1)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)
	print "Naive Bayes (scikit): accuracy:   %0.3f" % score
	print "-------------------------------------------"
	print "-------------------------------------------"
	errorNB[a] = 1.0 - score
	
	a += 1

print "NB  error: ",1.0-mean(errorNB)




