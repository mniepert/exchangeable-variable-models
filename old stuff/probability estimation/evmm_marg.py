import numpy as np
from operator import mul
import operator
import scipy
from collections import Counter
import sys, traceback

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
	partition = array([])
	smooth = 1.0
	nk = dict()
	sampleWeights = dict()
	laplace = 0.1

	# part is the partition in form of an array [0, 0, 1, 0, 1, 2, 3, 1] indicating column membership on (here: 4) partitions
	def __init__(self, data, part, weights):
		
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
		# stores the accumulated weights 
		self.sampleWeights = dict()

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
			configTuple = tuple(configs[i])
			#print configTuple
			# configs[i] stores the count for each block
			sWeight = self.sampleWeights.get(configTuple, -1.0)
			#print sWeight
			if sWeight > 0:
				sWeight += weights[i]
				#print sWeight
				self.sampleWeights[configTuple] = sWeight
			else:
				self.sampleWeights[configTuple] = weights[i]


		#print dummyCounter
		# number of configurations that have no occurence in the training data
		diff = prod - len(self.sampleWeights)

		#print "diff: ",diff

		#  the normalization constant for the probabilities of the block configurations
		self.smooth = float(sum(self.sampleWeights.values()))

		#print self.smooth

		# perform Laplacian smoothing -> add 1 count to each possible configuration
		# we do this only for the fully exchangeable component (splitNr == 1)
		self.smooth += diff*self.laplace
		for i in list(self.sampleWeights):
			self.sampleWeights[i] += self.laplace
			self.smooth += self.laplace

		#print self.sampleWeights

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
		sw = self.sampleWeights.get(x, -1.0)		
		if  sw > 0.0:
			currProb = float(sw)/self.smooth
			#print "No SMOOTH! -- ",currProb
		else:
			# if the configuration probability is zero and we are fully exchangeable, apply smoothing
			currProb = self.laplace/self.smooth

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
				self.nk[cnk] = tst
				currProb = currProb / tst

		return currProb


class IndComponent:

	comp = array([])
	splitNumbers = [0]
	partition = array([])

	# part is the partition in form of an array [0, 0, 1, 0, 1, 2, 3, 1] indicating column membership on (here: 4) partitions
	def __init__(self, data, weights):

		#print weights
		#print "------"

		# get the dimensions of the trainging data matrix
		m,n = data.shape

		# compute the priors from the training data: prob(x=1)
		msTemp = zeros(n,dtype=float)
		for i in arange(n):
			msTemp[i] = (float(sum(weights[data[:,i]==1])) + 0.1) / (float(sum(weights[data[:,i]<=1])) + 0.2)

		#print "marginals of component: ",msTemp


		# compute the blocks with identical marginal probability
		assign = zeros(n,dtype=int)
		countUnique = 0
		# counts the number of exchangeable blocks
		blockSizeCounter = 0
		# sorted marginals
		sortedMsTemp = sort(unique(msTemp))
		previousValue = sortedMsTemp[0]
		
		for j in arange(len(sortedMsTemp)):
			#print j
			if abs(previousValue-sortedMsTemp[j]) <= 0.0:
				assign[j] = countUnique
			else:
				countUnique += 1
				assign[j] = countUnique

			previousValue = sortedMsTemp[j]
			
		#print assign

		# the integers used in part to index the blocks (e.g.: [0, 2, 3])
		self.splitNumbers = unique(assign)
		# copy the partition indicator array to the class variable "partition"
		self.partition = assign

		self.comp = array([])

		for i in self.splitNumbers:
			mc,nc = data[:,self.partition==i].shape
			self.comp = append(self.comp, MComponent(data[:,self.partition==i], zeros(nc, dtype=int), weights))

		print "number of blocks: ",len(list(self.comp))


	def prob(self, data_point):
		# iterate over the number of blocks
		pr = 1.0
		for i in arange(len(self.comp)):
			pr = pr * self.comp[i].prob(data_point[self.partition==self.splitNumbers[i]])
				
		return pr


	def probLog(self, data_point):
		
		#print data_point

		# iterate over the number of blocks
		pr = 0.0
		for i in arange(len(self.comp)):
			pr = pr + log(self.comp[i].prob(data_point[self.partition==self.splitNumbers[i]]))

		return pr


# the name of the data set	
dataSetName = "book"

# load the training data
data = numpy.loadtxt(open(dataSetName+".ts.data","rb"),dtype=int,delimiter=",")

# get the dimensions of the trainging data matrix
m,n = data.shape

# compute the priors from the training data: prob(x=1)
ms = zeros(n,dtype=float)
for i in arange(n):
	ms[i] = mean(data[:,i])

# the number of mixture components (latent variable values)
numComponents = 10

initData = np.random.randint(2, size=(m, n))
initDataIndicator = np.random.randint(numComponents, size=(m,))

# comp are the mixture components
comp = array([])
for j in arange(numComponents):
	initAssign = zeros(m, dtype=float)
	initAssign[initDataIndicator==j] = 1.0
	# create a mixture component for the ith row having value '0'
	compTemp = IndComponent(initData, initAssign)
	#savetxt(dataSetName+str(j)+".shuffle.data", data, fmt='%c', delimiter=',')   # X is an array
	comp = append(comp, compTemp)

# class probabilities initialized to uniform probabilities
latentProb = ones(numComponents, dtype=float)
latentProb = latentProb/sum(latentProb)

print "latent class probability: ",latentProb

for c in arange(10):
	print "EM iteration: ",c
	# iterate over the training samples (all of them) an compute probability
	compPr = zeros(numComponents, dtype=float)
	weights = zeros((numComponents, m), dtype=float)
	# the E step
	for i in arange(m):
		probSum = 0.0
		for j in arange(numComponents):
			# probability (unnormalized) of the data point i for the component j
			prob = log(latentProb[j]) + comp[j].probLog(data[i])
			prob = exp(prob)
			weights[j][i] = prob
			# the sum of the probabilites (used for normalization)
			probSum += prob
			#print weights[j][i]
		
		#print " "
		#print weights
		#print probSum

		for j in arange(numComponents):
			# normalize the probabilities
			if probSum <= 0.0:
				weights[j][i] = 0.0
			else:
				weights[j][i] = weights[j][i] / probSum
			# aggregate the normalized probabilities
			compPr[j] += weights[j][i]
			#print weights[j][i],

		#print " ---- "

	# the M step
	# update the class priors
	latentProb = compPr/sum(compPr)

	# update the parameters of the mixture components
	# comp are the mixture components
	comp = array([])
	# run inference in the components and compute the new probabilities
	for j in arange(numComponents):
		# this indicates that we are using the fully exchangeable model
		#assign = zeros(n,dtype=int)
		# create a mixture component for the ith row having value '0'
		compTemp = IndComponent(data, weights[j])
		#print weights[j]
		#print " ---- "
		comp = append(comp, compTemp)
		#print weights[j]

	print "latent class probability: ",latentProb
	#print "---"


# load test data
data_test = numpy.loadtxt(open(dataSetName+".test.data","rb"),dtype=int,delimiter=",")

# dimensions of test data
mt,nt = data_test.shape

# compute the log-likelihood of the test data for the partial exchangeable model
testCounter = 0
sumn = 0.0
for x in data_test:
	testCounter += 1
	prob = 0.0
	for j in arange(numComponents):
		if latentProb[j] > 0.0:
			prob = prob + latentProb[j] * comp[j].prob(x)
	sumn = sumn + log(prob)

print testCounter
print "Mixture of independent EVMs: ",sumn / len(data_test)


