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
from scipy import stats
from sets import Set

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
	laplace = 0.1

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


print "Loading parity dataset:"

numSamples = 11000
numVariables = 1000

numCombinations = 10

errorEVM = zeros(numCombinations, dtype=float)
errorNB = zeros(numCombinations, dtype=float)
errorDT = zeros(numCombinations, dtype=float)
errorSVM = zeros(numCombinations, dtype=float)
errorkNN = zeros(numCombinations, dtype=float)

diffSumEVM_NB = zeros(numCombinations, dtype=float)
diffSumEVM_SVM = zeros(numCombinations, dtype=float)
diffSumEVM_DT = zeros(numCombinations, dtype=float)
diffSumEVM_kNN = zeros(numCombinations, dtype=float)

for a in arange(numCombinations):


	X_data = zeros((numSamples, numVariables), dtype=int)
	y_data = array([])

	for i in arange(numSamples):
		#print i
		if randint(1, 10) > 5:
			row = zeros(numVariables,dtype=int)
			for z in arange(randint(numVariables)):
				row[z] = 1
			shuffle(row)
			#row = np.random.randint(2, size=(numVariables))	
			count = count_nonzero(row)
			while mod(count, 2) < 1:
				row = zeros(numVariables,dtype=int)
				for z in arange(randint(numVariables)):
					row[z] = 1
				shuffle(row)
				#row = np.random.randint(2, size=(numVariables))	
				count = count_nonzero(row)

			X_data[i] = row
			y_data = append(y_data, 0)
		
		else:
			row = zeros(numVariables,dtype=int)
			for z in arange(randint(numVariables)):
				row[z] = 1
			shuffle(row)
			count = count_nonzero(row)
			while mod(count, 2) > 0:
				row = zeros(numVariables,dtype=int)
				for z in arange(randint(numVariables)):
					row[z] = 1
				shuffle(row)
				count = count_nonzero(row)
			#row = np.random.randint(2, size=(numVariables))	
			X_data[i] = row
			y_data = append(y_data, 1)
			

	#print y_data

	y_train, y_test = y_data[:10000], y_data[10000:]
	X_train, X_test = X_data[:10000], X_data[10000:]


	#vectorizer = CountVectorizer(binary=True)
	#X_train = vectorizer.fit_transform(X_train)
	#X_test = vectorizer.transform(X_test)

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
			#print "i: ",i,"   ",stats.ttest_ind(dataSet[i][:,0], dataSet[i][:,j], equal_var=False)
			msTemp[j] = (float(sum(dataSet[i][:,j]==1)) + 0.1) / (float(dataSet[i].shape[0]) + 0.2)
			
		dataSetMarg[i] = copy(msTemp)
		#print msTemp
	
	# array storing the independent exchangeable components
	comp = array([])
	# counts the number of exchangeable blocks
	blockCounter = 0
	# compute the blocks with identical marginal probability
	for i in arange(numOfClasses):
		assign = zeros(n,dtype=int)
		countUnique = 0
		msTemp = dataSetMarg[i]
		for j in unique(msTemp):
			for k in arange(n):
				if abs(msTemp[k]-j) <= 0.05:
					assign[k] = countUnique
					blockCounter += 1
				# maximum size of exchangeable sequence
				if blockCounter >= 1000:
					countUnique += 1
					blockCounter = 0
			countUnique += 1
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

	clf = DecisionTreeClassifier()
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

	clf = KNeighborsClassifier()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)
	print "5-NN: accuracy:   %0.3f" % score
	print "-------------------------------------------"
	errorkNN[a] = 1.0 - score
	diffSumEVM_kNN[a] = errorEVM[a] - errorkNN[a]


print "EVM error: ",mean(errorEVM)
print "NB  error: ",mean(errorNB)
print "DT  error: ",mean(errorDT)
print "SVM  error: ",mean(errorSVM)
print "kNN  error: ",mean(errorkNN)

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

print "EVM vs. kNN"
print mean(diffSumEVM_kNN)
print std(diffSumEVM_kNN)/(sqrt(len(diffSumEVM_kNN)))
