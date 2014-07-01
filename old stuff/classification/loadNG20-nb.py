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
				configs[i][j] = count_nonzero(data[i][part==self.splitNumbers[j]])

		# map the array of all configurations to a hash table of tuples
		configs_hash = map(tuple, configs)

		# count the frequencies of the different configurations
		self.c = Counter(configs_hash)

		# number of configurations that have no occurence in the training data
		diff = prod - len(self.c)

		#  the normalization constant for the probabilities of the block configurations
		self.smooth = float(sum(self.c.values()))

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

	def getNumberOfBlocks(self):
		return float(len(self.comp))


NG20Categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

NG20Combinations = list(combinations(NG20Categories, 2))
numCombinations = len(list(NG20Combinations))
print "Number of unique pairs: ",numCombinations

errorEVM = zeros(numCombinations, dtype=float)
errorNB = zeros(numCombinations, dtype=float)
errorDT = zeros(numCombinations, dtype=float)
errorSVM = zeros(numCombinations, dtype=float)
errorkNN = zeros(numCombinations, dtype=float)

diffSumEVM_NB = zeros(numCombinations, dtype=float)
diffSumEVM_SVM = zeros(numCombinations, dtype=float)
diffSumEVM_DT = zeros(numCombinations, dtype=float)
diffSumEVM_kNN = zeros(numCombinations, dtype=float)

blockSizeStatistics = zeros(2*numCombinations, dtype=float)

dataSizeStatisticsTrain = zeros(numCombinations, dtype=float)
dataSizeStatisticsTest = zeros(numCombinations, dtype=float)

varSizeStatisticsTrain = zeros(numCombinations, dtype=float)
varSizeStatisticsTest = zeros(numCombinations, dtype=float)

a = 0

for categories in NG20Combinations:

#for itv in arange(1):

	#categories = NG20Categories

	print "Loading 20 newsgroups dataset for categories:"
	print categories

	data_train_raw = fetch_20newsgroups(subset='train',  remove=('headers', 'footers', 'quotes'), categories=categories)
	print "%d documents" % len(data_train_raw.filenames)
	print "%d categories" % len(data_train_raw.target_names)

	data_test_raw = fetch_20newsgroups(subset='test',  remove=('headers', 'footers', 'quotes'), categories=categories)
	print "%d documents" % len(data_test_raw.filenames)
	print "%d categories" % len(data_test_raw.target_names)


	# split a training set and a test set
	y_train, y_test = data_train_raw.target, data_test_raw.target

	vectorizer = CountVectorizer(binary=True)
	X_train = vectorizer.fit_transform(data_train_raw.data)

	X_test = vectorizer.transform(data_test_raw.data)

	data_train = X_train.toarray()
	m,n = data_train.shape

	dataSizeStatisticsTrain[a] = float(m)
	varSizeStatisticsTrain[a] = float(n)

	clf = BernoulliNB(alpha=0.1)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)
	print "Naive Bayes (scikit): accuracy:   %0.3f" % score
	print "-------------------------------------------"
	print "-------------------------------------------"
	errorNB[a] = 1.0 - score
	diffSumEVM_NB[a] = errorEVM[a] - errorNB[a]

	#clf = tree.DecisionTreeClassifier()
	#clf.fit(X_train.toarray(), y_train)
	#pred = clf.predict(X_test.toarray())
	#score = metrics.accuracy_score(y_test, pred)
	#print "Decision tree: accuracy:   %0.3f" % score
	#print "-------------------------------------------"
	#print "-------------------------------------------"
	#errorDT[a] = 1.0 - score
	#diffSumEVM_DT[a] = errorEVM[a] - errorDT[a]

	#clf = LinearSVC()
	#clf.fit(X_train, y_train)
	#pred = clf.predict(X_test)
	#score = metrics.accuracy_score(y_test, pred)
	#print "SVM: accuracy:   %0.3f" % score
	#print "-------------------------------------------"
	#errorSVM[a] = 1.0 - score
	#diffSumEVM_SVM[a] = errorEVM[a] - errorSVM[a]

	#clf = KNeighborsClassifier()
	#clf.fit(X_train, y_train)
	#pred = clf.predict(X_test)
	#score = metrics.accuracy_score(y_test, pred)
	#print "5-NN: accuracy:   %0.3f" % score
	#print "-------------------------------------------"
	#errorkNN[a] = 1.0 - score
	#diffSumEVM_kNN[a] = errorEVM[a] - errorkNN[a]

	a += 1


print "EVM error: ",1.0-mean(errorEVM)
print "NB  error: ",1.0-mean(errorNB)
print "DT  error: ",1.0-mean(errorDT)
print "SVM  error: ",1.0-mean(errorSVM)
print "5-NN  error: ",1.0-mean(errorkNN)

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

print "EVM vs. 5-NN"
print mean(diffSumEVM_kNN)
print std(diffSumEVM_kNN)/(sqrt(len(diffSumEVM_kNN)))

print "Train data, samples & vars: ", mean(dataSizeStatisticsTrain), "; ", mean(varSizeStatisticsTrain)
print "Test data, samples & vars: ", mean(dataSizeStatisticsTest), "; ", mean(varSizeStatisticsTest)

print "Blocks mean/std:", mean(blockSizeStatistics), "; ", std(blockSizeStatistics)


