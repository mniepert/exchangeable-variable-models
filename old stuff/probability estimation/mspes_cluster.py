import numpy as np
from operator import mul
import operator
import scipy
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from decimal import *


def  n_take_k(n,r):
    ''' calculate nCr - the binomial coefficient
    >>> comb(3,2)
    3
    >>> comb(9,4)
    126
    >>> comb(9,6)
    84
    >>> comb(20,14)
    38760'''
 
    if r > n-r:  # for smaller intermediate values
        r = n-r
    return int( reduce( mul, range((n-r+1), n+1), 1) /
      reduce( mul, range(1,r+1), 1) )


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


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

		# stores the possible binomial coefficients
		self.nk = dict()

		#the number of configurations
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

		#print self.c

		# number of configurations that have no occurence
		diff = prod - len(self.c)

		print "diff: ",diff

		#  the normalization constant for the probabilities of the block configurations
		self.smooth = float(sum(self.c.values()))#+float(diff)

		# perform smoothing -> add 1 count to each possible configuration
		# we do this only for the full exchangeable component (==1)
		if self.splitNr == 1:
			self.smooth += diff
			for i in arange(len(self.c)):
				self.c[i] += 1
				self.smooth += 1

		#print sort(list(self.c), axis=0)

	def prob(self, data_point):

		# the vector representing the projection of the data point to the exchangeable blocks
		configs_test = zeros((self.splitNr,), dtype=int)

		# iterate over the number of blocks
		for i in arange(self.splitNr):
			configs_test[i] = count_nonzero(data_point[self.partition==self.splitNumbers[i]])

		#print data_point
		#print configs_test

		# convert the array to a tuple
		x = tuple(configs_test)
		
		# look up the probability of the given block configuration		
		if self.c[x] > 0:
			currProb = float(self.c[x])/self.smooth
		else:
			#print bincount(self.partition)
			#print x
			if self.splitNr == 1:
				currProb = 1.0/self.smooth
			else:			
				return 0.0
			#currProb = 1.0/self.smooth
			#return (1.0/float(pow(2.0,len(data_point))))#/self.smooth

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

	#the data point has three values: 0,1 indicating non-query and query variables
	def cond_prob(self, data_point, query):

		# count the occurrences of the values in blocks
		fc = zeros((self.splitNr,), dtype=int)

		# lower bound of number of 1s in the ith block
		lb1 = zeros((self.splitNr,), dtype=int)

		# upper bound of number of 1s in the ith block
		ub1 = zeros((self.splitNr,), dtype=int)

		# stores the size of the blocks of this particular partition
		blockSize = zeros((self.splitNr,), dtype=int)

		# stores the number of query variables of each block
		qVarsBlock = zeros((self.splitNr,), dtype=int)
		
		# iterate over the number of blocks
		for i in arange(self.splitNr):
			# cut out the current block
			currentBlock = data_point[self.partition==self.splitNumbers[i]]
			# count the occurrences of non-zero values in the ith block (the projection value for this block)
			fc[i] = count_nonzero(currentBlock)
			# cut the current block out of the query variables
			currentQueryBlock = query[self.partition==self.splitNumbers[i]]
			# count the number of 1s that are evidence variables (minimum value of 1s)
			lowerBound = (currentBlock > currentQueryBlock).astype(int)
			lb1[i] = count_nonzero(lowerBound)
			# the number of query variables in this block			
			qVarsBlock[i] = count_nonzero(currentQueryBlock)
			# count the number of evidence 1s plus number of query variables in block			
			ub1[i] = lb1[i] + qVarsBlock[i]
			# determine the block size
			blockSize[i] = len(currentBlock)
		
		#print fc
		#print lb1
		#print ub1
		#print "dp: ",data_point
		#print "qr: ",query
		#print "tp: ",tuple([lb1, ub1])
				
		# build arrays each ranging over one dimension		
		cartesianArray = list()
		for i in arange(self.splitNr):
			cartesianArray.append(arange(lb1[i], ub1[i]+1))
			
		#print cartesianArray

		# compute the set of arrays representing all possible configurations over the non-evidence variables
		possibleConfigs = cartesian(cartesianArray)

		#print possibleConfigs

		# convert array to set of tuples
		data_hash = map(tuple, possibleConfigs)
		#print "pc: ",data_hash

		######### We now compute the probability of the entire tuple ###################
		# the numerator
		numerator = 0.0
		# look up the probability
		x = tuple(fc)
		if self.c[x] > 0:
			numerator = float(self.c[x])/self.smooth
		else:
			if self.splitNr == 1:
				numerator = 1.0/self.smooth
			else:			
				return 0.0	
			#numerator = 1.0/self.smooth

		# normalize by the number of configuration represented by this particular block configuration
		for i in arange(self.splitNr):
			numerator = numerator / n_take_k(blockSize[i], fc[i])
	
		#print len(data_hash)

		# here we sum over all possible configurations given the non-evidence variables
		# sum of the probabilites (the denominator)
		sumDe = 0.0
		# iterate again over all of the blocks
		for x in data_hash:
			# look up probability of this particular configuration
			if self.c[x] > 0:
				currProb = float(self.c[x])/self.smooth
			else:
				if self.splitNr == 1:
					currProb = 1.0/self.smooth
				else:			
					currProb = 0.0
					continue

			# normalize by the number of configuration represented by this particular block configuration
			for i in arange(self.splitNr):
				nvalue = blockSize[i]
				kvalue = x[i]
				cnk = tuple([nvalue, kvalue])
				tst = self.nk.get(cnk, False)
				if tst:
					cnk2 = tuple([qVarsBlock[i], kvalue-lb1[i]])
					tst2 = self.nk.get(cnk2, False)
					if tst2:
						currProb = (float(tst2) / float(tst)) * currProb
					else:
						tst2 = n_take_k(qVarsBlock[i], kvalue-lb1[i])
						currProb = (float(tst2) / float(tst)) * currProb
						self.nk.setdefault(cnk2, tst2)
				else:
					tst = n_take_k(nvalue, kvalue)
					self.nk.setdefault(cnk, tst)
					cnk2 = tuple([qVarsBlock[i], kvalue-lb1[i]])
					tst2 = self.nk.get(cnk2, False)
					if tst2:
						currProb = (float(tst2) / float(tst)) * currProb
					else:
						tst2 = n_take_k(qVarsBlock[i], kvalue-lb1[i])
						currProb = (float(tst2) / float(tst)) * currProb
						self.nk.setdefault(cnk2, tst2)

			# add the probability to the overall sum of probabilities
			sumDe = sumDe + currProb

		return numerator/sumDe


	
dataSetName = "book"
numQueryVariables = 50

# load the training data
data = numpy.loadtxt(open(dataSetName+".ts.data","rb"),dtype=int,delimiter=",")

# get the dimensions of the trainging data matrix
m,n = data.shape

# compute the marginal probabilities of the involved variables
ms = zeros(n,dtype=float)
for i in arange(n):
	ms[i] = average(data[:,i])



################################################################################
############ Leverage validation data to learn the mixture weights #############
#################### HARD AND SOFT EM ##########################################
################################################################################

# load validation data
data_valid = numpy.loadtxt(open(dataSetName+".valid.data","rb"),dtype=int,delimiter=",")

# get dimensions of validation data matrix
mv,nv = data_valid.shape

# the number of mixture components
numMixtures = 50


# load the precomputed distance matrix 
dmatrix = loadtxt(open(dataSetName+"_matrix.out","rb"),dtype=float,delimiter=",")
clustering = linkage(dmatrix)

comp = array([])
for i in arange(numMixtures):
	
	# compute the clustering with at most i+1 clusters
	assign = fcluster(clustering, i+1, criterion='maxclust')
	#assign = zeros(nv,dtype=int)
	#assign[0] = 1
	#assign[1] = 1
	#if i == 0:
	#	assign = zeros(nv,dtype=int)
	#print assign
	# create a mixture component based on the clustering
	celem = MComponent(data, assign)
	# add the mixture component to the mixture
	comp = append(comp, celem)
	print "Mixture component: ",i


maxCounter = ones(numMixtures)
mixtureProb = zeros(numMixtures, dtype=float)
mixtureProb = map(float, mixtureProb + 1.0/float(numMixtures))
#iterate over the validation data points and compute the probability for each component
for x in data_valid:
	maxProb = 0.0
	maxComp = 1
	for i in arange(numMixtures):
		mixtureProb[i] = mixtureProb[i]*comp[i].prob(x)
		if mixtureProb[i] > maxProb:
			maxProb = mixtureProb[i]
			maxComp = i

	maxCounter[maxComp] += 1
	mixtureProb = map(float, maxCounter/sum(maxCounter))

# sort the probabilities of the components (ascending)
agr = argsort(mixtureProb)
pw = take(mixtureProb, agr)
# sort the list with the actual components accordingly
comp = take(comp, agr)

# reverse the order so as to have a decreasing order
mixtureProb = mixtureProb[::-1]
comp = comp[::-1]

#print pw

# the number of top components we want to retrain
collectTopComponents = 20

maxCounter = ones(collectTopComponents)
mixtureProb = zeros(collectTopComponents, dtype=float)
mixtureProb = map(float, mixtureProb + 1.0/float(collectTopComponents))
#iterate over the validation data points and compute the probability for each component
for x in data_valid:
	maxProb = 0.0
	maxComp = 1
	for i in arange(collectTopComponents):
		mixtureProb[i] = mixtureProb[i]*comp[i].prob(x)
		if mixtureProb[i] > maxProb:
			maxProb = mixtureProb[i]
			maxComp = i

	maxCounter[maxComp] += 1
	mixtureProb = map(float, maxCounter/sum(maxCounter))
	#print mixtureProb

print "Mixture weights EM: ",mixtureProb

############################################################################
#################### EVALUATION ############################################
############################################################################

# load test data
data_test = numpy.loadtxt(open(dataSetName+".test.data","rb"),dtype=int,delimiter=",")

# dimensions of test data
mt,nt = data_test.shape

# compute the log-likelihood of the model that assumes independence between all variables
# go through the rows of the test data matrix and compute the log-likelihood of the test data
testCounter = 0
sum_ind = 0.0
for i in arange(mt):
	testCounter += 1
	pr = 1.0
	for j in arange(nt):
		pr = pr*((1.0-data_test[i][j])*(1.0-ms[j])+data_test[i][j]*ms[j])
	if pr > 0:
		sum_ind = sum_ind + log(pr)
	else:
		print "ind 0"
		sum_ind = sum_ind + log(1.0/float(pow(2.0,nt)))

print testCounter
print "Independent model: ",sum_ind / mt

# compute the log-likelihood of the test data for the partial exchangeable model
testCounter = 0
sumn = 0.0
for x in data_test:
	testCounter += 1
	prob = 0.0
	for i in arange(collectTopComponents):
		if mixtureProb[i] > 0.0:
			prob = prob + mixtureProb[i] * comp[i].prob(x)
	sumn = sumn + log(prob)

print testCounter
print "Mixture of smoothed partially exchangeable sequences: ",sumn / len(data_test)

# set the array that indicates evidence/non-evidence
assign = zeros(nt, dtype=int)
for i in arange(numQueryVariables):
	assign[i] = 1

# compute the conditional log-likelihood of the model that assumes independence between all variables
# go through the rows of the test data matrix and compute the log-likelihood of the test data
testCounter = 0
sum_ind = 0.0
sumn = 0.0
for x in data_test:
	testCounter += 1
	progress = 100*float(testCounter)/float(mt)
	sys.stdout.write("\r%d%%" %progress)

	# shuffle the evidence/query indicator array
	shuffle(assign)

	prob = 0.0
	for i in arange(collectTopComponents):
		if mixtureProb[i] > 0.0:
			prob = prob + mixtureProb[i] * comp[i].cond_prob(x, assign)
	sumn = sumn + log(prob)

	# here we compute the conditional log-likelihood for the fully independent model
	pr = 1.0
	for j in arange(nt):
		pr = pr*((1.0-x[j])*(1.0-ms[j])+x[j]*ms[j])

	prd = 1.0
	for j in arange(nt):
		if assign[j] == 0:
			prd = prd*((1.0-x[j])*(1.0-ms[j])+x[j]*ms[j])
	
	if (pr > prd):
		print "weird!"
	
	if pr > 0 and prd > 0:
		sum_ind = sum_ind + log(pr/prd)
	else:
		sum_ind = sum_ind + log(1.0/float(pow(2.0,nt)))

print ""
print testCounter
print "Independent model: ",(sum_ind / float(len(data_test))) / float(numQueryVariables)
print "\nMixture of smoothed partially exchangeable sequences: ",(sumn / float(len(data_test))) / float(numQueryVariables)
