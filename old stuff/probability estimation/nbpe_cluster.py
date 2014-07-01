import numpy as np
from operator import mul
import operator
import scipy
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

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

	# part is the partition in form of an array [0, 0, 1, 0, 1, 2, 3, 1] indicating column membership on (here: 4) partitions
	def __init__(self, data, part):
		
		# get dimensions of the data matrix
		m,n = data.shape

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

		#print "diff: ",diff

		#  the normalization constant for the probabilities of the block configurations
		self.smooth = float(sum(self.c.values()))

		# perform Laplacian smoothing -> add 1 count to each possible configuration
		# we do this only for the fully exchangeable component (splitNr == 1)
		if self.splitNr >= 1:
			self.smooth += diff
			for i in list(self.c):
				self.c[i] += 1
				self.smooth += 1

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
			currProb = float(self.c[x])/self.smooth
		else:
			# if the configuration probability is zero and we are fully exchangeable, apply smoothing
			if self.splitNr >= 1:
				currProb = 1.0/self.smooth
			# if the configuration probability is zero and we are *not* fully exchangeable, return 0.0
			else:
				return 0.0

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

		return currProb


# the name of the data set	
dataSetName = "baudio"

# load the training data
data = numpy.loadtxt(open(dataSetName+".ts.data","rb"),dtype=int,delimiter=",")

# get the dimensions of the trainging data matrix
m,n = data.shape

# compute the priors from the training data: prob(x=1)
prior = zeros(n,dtype=float)
for i in arange(n):
	prior[i] = mean(data[:,i])


print prior

A = argsort(prior)
prior = sort(prior)

data = data[arange(data.shape[0])[:,newaxis],A]

resultNB = zeros(n, dtype=float)
resultNBE = zeros(n, dtype=float)
resultBL = zeros(n, dtype=float)

resultCLLNB = zeros(n, dtype=float)
resultCLLNBE = zeros(n, dtype=float)


# iterate over all variables in the data set
for position in arange(n):

	print "Processing column ",position," out of ",n,"" 
	print "1-Prior: ",1.0-prior[position]

	# get the matrix where the position column's value is 0
	data0 = data[data[:,position]==0]

	# get the matrix where the position column's value is 1
	data1 = data[data[:,position]==1]

	# compute the marginal probabilities of the involved variables
	ms0 = zeros(n,dtype=float)
	ms1 = zeros(n,dtype=float)
	ms1t = zeros(n,dtype=float)
	for i in arange(n):
		ms0[i] = (float(sum(data0[:,i]==1)) + 1.0) / (float(data0.shape[0])+2.0)
		ms1[i] = (float(sum(data1[:,i]==1)) + 1.0) / (float(data1.shape[0])+2.0)
		#ms1t[i] = mean(data0[:,i])

	#print ms0
	#print ms0
	#print ms1t

	# this is used to extract the submatrix (the training data minus the target column)
	target = ones(n, dtype=int)
	target[position] = 0

	# load validation data to compute log-likelihood
	data_valid = numpy.loadtxt(open(dataSetName+".valid.data","rb"),dtype=float,delimiter=",")
	data_valid = data_valid[arange(data_valid.shape[0])[:,newaxis],A]

	# dimensions of test data
	mv,nv = data_valid.shape

	# load the precomputed distance matrix 
	dmatrix = loadtxt(open(dataSetName+"_matrix.out","rb"),dtype=float,delimiter=",")
	clustering = linkage(dmatrix)
	# compute the clustering with at most i+1 clusters
	assign = fcluster(clustering, 5, criterion='maxclust')

	assign = assign[target==1]

	# this indicates that we are using the fully exchangeable model
	#assign = zeros(n-1,dtype=int)
	# create the best partial exchangeable sequence
	#assign[:iMax] = 1
	#print assign
	# create a mixture component for the ith row having value '0'
	comp0 = MComponent(data0[:,target==1], assign)
	
	# this indicates that we are using the fully exchangeable model
	#assign = zeros(n-1,dtype=int)
	# create the best partial exchangeable sequence
	#assign[:jMax] = 1
	#print assign
	# create a mixture component for the ith row having value '1'
	comp1 = MComponent(data1[:,target==1], assign)
	

	############################################################################
	#################### EVALUATION ############################################
	############################################################################

	# load test data
	data_test = numpy.loadtxt(open(dataSetName+".test.data","rb"),dtype=int,delimiter=",")
	data_test = data_test[arange(data_test.shape[0])[:,newaxis],A]

	# dimensions of test data
	mt,nt = data_test.shape

	# compute the accuracy of the naive Bayes model (independence of variables given the class) on the test data
	cllSum = 0.0
	correctCounter = 0
	for i in arange(mt):

		pr0 = 1.0
		pr1 = 1.0
		for j in arange(nt):
			if j != position:
				pr0 = pr0*((1.0-data_test[i][j])*(1.0-ms0[j])+data_test[i][j]*ms0[j])
				pr1 = pr1*((1.0-data_test[i][j])*(1.0-ms1[j])+data_test[i][j]*ms1[j])

		pr0 = pr0 * (1.0-prior[position])
		pr1 = pr1 * prior[position]

		if pr0 >= pr1 and data_test[i][position]==0:
			correctCounter += 1
		elif pr0 < pr1 and data_test[i][position]==1:	
			correctCounter += 1

		if data_test[i][position]==0:
			if pr0 <= 0.0:# to small --> was made zero
				cllSum = cllSum + log(0.00000000000000001)
			else:
				cllSum = cllSum + log(pr0/(pr0+pr1))
			#print pr0/(pr0+pr1), "        ",pr0,"            ",pr1
		else:
			if pr1 == 0.0 and pr0==0.0:
				cllSum += 0.0
			elif pr1 <= 0.0:# to small --> was made zero
				cllSum = cllSum + log(0.00000000000000001)
			else:
				cllSum = cllSum + log(pr1/(pr0+pr1))
			#print pr1/(pr0+pr1), "        ",pr1,"            ",pr0
			#print data_test[i]


	resultNB[position] = 1.0 - (float(correctCounter) / float(mt))
	resultCLLNB[position] = cllSum/float(mt)
	print "Independent model; error: ",1.0-float(correctCounter) / float(mt),"; CLL: ",cllSum/float(mt)


	# compute the accuracy for the exchangeable Naive Bayes model (exchangeability of variables given the class) on the test data
	cllSum = 0.0
	correctCounter = 0
	for i in arange(mt):
		
		pr0 = comp0.prob(data_test[i,target==1])
		pr1 = comp1.prob(data_test[i,target==1])

		pr0 = pr0 * (1.0-prior[position])
		pr1 = pr1 * prior[position]

		if pr0 >= pr1 and data_test[i][position]==0:
			correctCounter += 1
		elif pr0 < pr1 and data_test[i][position]==1:	
			correctCounter += 1

		if data_test[i][position]==0:
			cllSum = cllSum + log(pr0/(pr0+pr1))
			#print pr0/(pr0+pr1), "        ",pr0,"            ",pr1
		else:
			cllSum = cllSum + log(pr1/(pr0+pr1))
			#print pr1/(pr0+pr1), "        ",pr1,"            ",pr0

	
	resultNBE[position] = 1.0 - (float(correctCounter) / float(mt))
	resultCLLNBE[position] = cllSum/float(mt)
	print "Exchangeable model; error: ",1.0-float(correctCounter) / float(mt),"; CLL: ",cllSum/float(mt)


	# compute the accuracy of the majority baseline
	if prior[position] < 0.5:
		resultBL[position] = prior[position]
	else:
		resultBL[position] = (1.0-prior[position])


print dataSetName
#print "Baseline; mean: ",mean(resultBL), "  standard Deviation: ",std(resultBL)
#print "Independent model; mean: ",mean(resultNB), "  standard Deviation: ",std(resultNB)
#print "Exchangeable model: mean: ",mean(resultNBE), "  standard Deviation: ",std(resultNBE)
#print "Mean of differences (BL<->NB): ",mean(resultBL - resultNB)
#print "StdDev of differences(BL<->NB): ",std(resultBL - resultNB)/(sqrt(len(resultNB)))
print "Independent model (error); mean: ",mean(resultNB), "  standard Deviation: ",std(resultNB)
print "Exchangeable model (error): mean: ",mean(resultNBE), "  standard Deviation: ",std(resultNBE)

print "CLL..."

print "Independent model (CLL); mean: ",mean(resultCLLNB), "  standard Deviation: ",std(resultCLLNB)
print "Exchangeable model (CLL): mean: ",mean(resultCLLNBE), "  standard Deviation: ",std(resultCLLNBE)
print "Mean of differences (CLL: NB<->NBE): ",mean(-resultCLLNB + resultCLLNBE)
print "StdDev of differences(CLL: NB<->NBE): ",std(-resultCLLNB + resultCLLNBE)/(sqrt(len(resultCLLNB)))

