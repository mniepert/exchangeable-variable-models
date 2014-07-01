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
from sklearn.datasets import fetch_mldata, load_files
from scipy.cluster.vq import vq, kmeans2, whiten
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# computes the binomial coefficient
def  n_take_k(n,r):
  
    if r > n-r:  # for smaller intermediate values
        r = n-r
    return int( reduce( mul, range((n-r+1), n+1), 1) /
      reduce( mul, range(1,r+1), 1) )


# class representing one mixture component
class MComponent:
	splitNr = 2
	splitNumbers = [0]
	c = Counter([])
	partition = array([])
	smooth = 1.0
	nk = dict()
	numVariables = 1
	laplace = 1.0

	# part is the partition in form of an array [0, 0, 1, 0, 1, 2, 3, 1] indicating column membership on (here: 4) partitions
	def __init__(self, data, part):
		
		# get dimensions of the data matrix
		m,n = data.shape

		# number of variables
		self.numVariables = n

		# compute number of blocks
		self.splitNr = len(unique(part))
		# copy the partition indicator array to the class variable "partition"
		self.partition = part
		# the integers used in part to index the blocks (e.g.: [0, 2, 3])
		self.splitNumbers = unique(part)
		
		#build an array of arrays that stores the probs
		configs = zeros((m, self.splitNr), dtype=int)

		# stores the possible binomial coefficients (caching)
		self.nk = dict()

		#compute the number of all possible configurations
		prod = 1.0
		for i in self.splitNumbers:
			blockSize = len(self.partition[part==i])
			#print "blockSize: ",blockSize
			prod = prod * (blockSize+1)

		# go through all rows of the data table and store number of configurations
		for i in arange(m):
			for j in arange(self.splitNr):
				#print "--------------"
				#print data[i]
				#print data[i][part==j]
				configs[i][j] = count_nonzero(data[i][part==self.splitNumbers[j]])
				#print "--------------"

		# map the array of all configurations to a hash table of tuples
		configs_hash = map(tuple, configs)

		# count the frequencies of the different configurations
		self.c = Counter(configs_hash)

		# number of configurations that have no occurence in the training data
		diff = prod - len(self.c)

		#print "prod: ",prod, "; diff: ",diff

		#  the normalization constant for the probabilities of the block configurations
		self.smooth = float(sum(self.c.values()))

		#self.laplace = 1.0 / (diff+1.0)

		# perform Laplacian smoothing -> add 1 count to each possible configuration
		# we do this only for the fully exchangeable component (splitNr == 1)
		self.smooth += diff*self.laplace
		for i in list(self.c):
			self.c[i] += self.laplace
			self.smooth += self.laplace

	# returns the probability of one particular configuration (here: conditional probability)
	def prob(self, data_point):

		# the vector representing the projection of the data point to the exchangeable blocks
		configs_test = zeros((self.splitNr,), dtype=int)

		# iterate over the number of blocks
		for i in arange(self.splitNr):
			configs_test[i] = count_nonzero(data_point[self.partition==self.splitNumbers[i]])

		# convert the array to a tuple (required for the look-up in the Counter structure)
		x = tuple(configs_test)
		
		# look up the probability of the given block configuration		
		if self.c[x] > 0:
			currProb = float(self.c[x])
		else:
			currProb = self.laplace
		
		# normalize by the number of configuration represented by this particular block configuration
		for i in arange(self.splitNr):
			nvalue = len(self.partition[self.partition==self.splitNumbers[i]])
			kvalue = configs_test[i]
			cnk = tuple([nvalue, kvalue])
			tst = self.nk.get(cnk, False)
			if tst:
				currProb = currProb / tst
			else:
				tst = n_take_k(nvalue, kvalue)
				self.nk.setdefault(cnk, tst)
				currProb = currProb / tst
		
		currProb = currProb / self.smooth

		return float(currProb)


class IndComponent:

	comp = array([])
	splitNumbers = [0]
	partition = array([])

	# part is the partition in form of an array [0, 0, 1, 0, 1, 2, 3, 1] indicating column membership on (here: 4) partitions
	def __init__(self, data, part):

		# the integers used in part to index the blocks (e.g.: [0, 2, 3])
		self.splitNumbers = unique(part)
		# copy the partition indicator array to the class variable "partition"
		self.partition = part

		self.comp = array([])

		for i in self.splitNumbers:
			mc,nc = data[:,part==i].shape
			self.comp = append(self.comp, MComponent(data[:,part==i], zeros(nc, dtype=int) ) )

		print "Number of exchangeable components: ",len(self.comp)

	def prob(self, data_point):
		
		#print data_point

		# iterate over the number of blocks
		pr = 1.0
		for i in arange(len(self.comp)):
			#print i
			#print data_point[self.partition==self.splitNumbers[i]]
			pr = pr * self.comp[i].prob(data_point[self.partition==self.splitNumbers[i]])

		return pr


	def probLog(self, data_point):
		
		#print data_point

		# iterate over the number of blocks
		pr = 0.0
		for i in arange(len(self.comp)):
			#print i
			#print data_point[self.partition==self.splitNumbers[i]]
			pr = pr + log(self.comp[i].prob(data_point[self.partition==self.splitNumbers[i]]))

		return pr


print "Loading random dataset:"

numCombinations = 10

errorEVM = zeros(numCombinations, dtype=float)
errorNB = zeros(numCombinations, dtype=float)
errorDT = zeros(numCombinations, dtype=float)
errorSVM = zeros(numCombinations, dtype=float)
errorSGD = zeros(numCombinations, dtype=float)

diffSumEVM_NB = zeros(numCombinations, dtype=float)
diffSumEVM_SVM = zeros(numCombinations, dtype=float)
diffSumEVM_DT = zeros(numCombinations, dtype=float)
diffSumEVM_SGD = zeros(numCombinations, dtype=float)

for a in arange(numCombinations):

	X_both,y_both = datasets.make_classification(n_samples=5000, n_features=10000, n_informative=10, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

	binarizer = Binarizer().fit(X_both)
	X_both = binarizer.transform(X_both)

	# split a training set and a test set
	y_train, y_test = y_both[:4000], y_both[4000:]

	X_train = X_both[:4000]
	X_test =  X_both[4000:]

	print "X_train: ",X_train.shape
	print "X_test: ",X_test.shape

	#ch2 = SelectKBest(chi2, 10000)
	#X_train = ch2.fit_transform(X_train, y_train)
	#X_test = ch2.transform(X_test)

	data_train = X_train
	m,n = data_train.shape

	print m," ",n

	# compute the priors from the training data: prob(x=1)
	prior = zeros(n,dtype=float)
	for i in arange(n):
		prior[i] = mean(data_train[:,i])

	#print prior

	dataSet = dict()
	dataSetMarg = dict()

	for i in unique(y_train):
		dataSet[i] = data_train[y_train==i]

	numOfClasses = len(unique(y_train))
	print numOfClasses

	# array storing the independent exchangeable components
	comp = array([])

	# compute the marginals for each of the class labels
	for i in arange(numOfClasses):
		msTemp = zeros(n, dtype=float)
		for j in arange(n):
			msTemp[j] = (float(sum(dataSet[i][:,j]==1)) + 0.1) / (float(dataSet[i].shape[0]) + 0.2)
		dataSetMarg[i] = copy(msTemp)
	
		#print dataSetMarg[i]

		# compute the blocks with identical marginal probability
		assign = zeros(n,dtype=int)
		countUnique = 0
		# counts the number of exchangeable blocks
		blockSizeCounter = 0
		# sorted marginals
		sortedMsTempArg = argsort(msTemp)
		sortedMsTemp = sort(msTemp)
		previousValue = sortedMsTemp[0]
		
		for j in arange(len(sortedMsTemp)):
			#print j
			if abs(previousValue-sortedMsTemp[j]) <= 0.0001:
				assign[sortedMsTempArg[j]] = countUnique
				blockSizeCounter += 1
			else:
				countUnique += 1
				assign[sortedMsTempArg[j]] = countUnique
				blockSizeCounter = 0
			previousValue = sortedMsTemp[j]

		comp = append(comp, IndComponent(dataSet[i], assign))


	############################################################################
	#################### EVALUATION ############################################
	############################################################################


	# load test data
	data_test = X_test

	#print data_test

	# dimensions of test data
	mt,nt = data_test.shape

	# compute the accuracy for the exchangeable Naive Bayes model (exchangeability of variables given the class) on the test data
	correctCounter = 0
	for i in arange(mt):
	
		pr = zeros(numOfClasses, dtype=float)	
		for j in arange(numOfClasses):
			pr[j] = comp[j].probLog(data_test[i])
			pr[j] = pr[j] + log(float(sum(y_train==j)) / float(len(y_train)))

		#print pr

		if y_test[i]==argmax(pr):
			correctCounter += 1

	errorEVM[a] = 1.0 - float(correctCounter) / float(mt)
	score = float(correctCounter) / float(mt)
	print "-------------------------------------------"
	print "-------------------------------------------"
	print "Exchangeable variable model (accuracy): %0.3f" % score
	print "-------------------------------------------"
	print "-------------------------------------------"
	

	clf = BernoulliNB()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)
	print "Naive Bayes (scikit): accuracy:   %0.3f" % score
	print "-------------------------------------------"
	print "-------------------------------------------"
	errorNB[a] = 1.0 - score
	diffSumEVM_NB[a] = errorEVM[a] - errorNB[a]

	clf = tree.DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)
	print "Decision tree: accuracy:   %0.3f" % score
	print "-------------------------------------------"
	print "-------------------------------------------"
	errorDT[a] = 1.0 - score
	diffSumEVM_DT[a] = errorEVM[a] - errorDT[a]

	clf = LinearSVC()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)
	print "SVM: accuracy:   %0.3f" % score
	print "-------------------------------------------"
	errorSVM[a] = 1.0 - score
	diffSumEVM_SVM[a] = errorEVM[a] - errorSVM[a]

	clf = SGDClassifier()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)
	print "SGD: accuracy:   %0.3f" % score
	print "-------------------------------------------"
	errorSGD[a] = 1.0 - score
	diffSumEVM_SGD[a] = errorEVM[a] - errorSGD[a]

print "EVM error: ",mean(errorEVM)
print "NB  error: ",mean(errorNB)
print "DT  error: ",mean(errorDT)
print "SVM  error: ",mean(errorSVM)
print "SGD  error: ",mean(errorSGD)

print " "

print "EVM vs. NB"
print mean(diffSumEVM_NB)
print std(diffSumEVM_NB)/(sqrt(len(diffSumEVM_NB)))

print "EVM vs. DT"
print mean(diffSumEVM_DT)
print std(diffSumEVM_DT)/(sqrt(len(diffSumEVM_DT)))

print "EVM vs. SVM"
print mean(diffSumEVM_SVM)
print std(diffSumEVM_SVM)/(sqrt(len(diffSumEVM_SVM)))

print "EVM vs. SGD"
print mean(diffSumEVM_SGD)
print std(diffSumEVM_SGD)/(sqrt(len(diffSumEVM_SGD)))