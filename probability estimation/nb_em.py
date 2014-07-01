import numpy as np
from operator import mul
import operator
import scipy
from collections import Counter
import sys, traceback

# class representing one mixture component
class MComponent:
	size = 1
	normalize = 1.0
	sampleWeights = dict()
	laplace = 0.1

	# initialize a variable of the naive bayes model
	def __init__(self, data, weights):
		
		# get dimensions of the data matrix
		m,n = data.shape

		# stores the accumulated weights 
		self.sampleWeights = dict()
		# size of the block
		self.size = n

		# go through all rows of the data table and store number of configurations
		for i in arange(m):
			configs = count_nonzero(data[i])
			#print configTuple
			# configs[i] stores the count for each block
			sWeight = self.sampleWeights.get(configs, -1.0)
			#print sWeight
			if sWeight > 0:
				sWeight += weights[i]
				#print sWeight
				self.sampleWeights[configs] = sWeight
			else:
				self.sampleWeights[configs] = weights[i]


		#print dummyCounter
		# number of configurations that have no occurence in the training data
		diff = n - len(self.sampleWeights) + 1

		#  the normalization constant for the probabilities of the block configurations
		self.normalize = float(sum(self.sampleWeights.values()))

		# perform Laplacian smoothing -> add 1 count to each possible configuration
		# we do this only for the fully exchangeable component (splitNr == 1)
		self.normalize += diff*self.laplace
		for i in list(self.sampleWeights):
			self.sampleWeights[i] += self.laplace
			self.normalize += self.laplace

		#print self.sampleWeights

	# returns the probability of one particular configuration (here: conditional probability)
	def prob(self, data_point):

		# iterate over the number of blocks
		configs_test = count_nonzero(data_point)

		# look up the probability of the given block configuration
		sw = self.sampleWeights.get(configs_test, -1.0)		
		if  sw > 0.0:
			currProb = float(sw)/self.normalize
		else:
			# apply laplace smoothing
			currProb = self.laplace/self.normalize

		return currProb


class IndComponent:

	comp = array([])
	splitNumbers = [0]
	partition = array([])

	# part is the partition in form of an array [0, 0, 1, 0, 1, 2, 3, 1] indicating column membership on (here: 4) partitions
	def __init__(self, data, weights):

		# get the dimensions of the trainging data matrix
		m,n = data.shape

		# naive bayes model
		assign = arange(n)
		
		# the integers used in part to index the blocks (e.g.: [0, 2, 3])
		self.splitNumbers = unique(assign)
		# copy the partition indicator array to the class variable "partition"
		self.partition = assign
		# initialize array of exchangeable components
		self.comp = array([])

		for i in self.splitNumbers:
			self.comp = append(self.comp, MComponent(data[:,self.partition==i], weights))

		#print "number of blocks: ",len(list(self.comp))


	def prob(self, data_point):
		# iterate over the number of blocks
		pr = 1.0
		for i in arange(len(self.comp)):
			pr = pr * self.comp[i].prob(data_point[i])
				
		return pr


	def probLog(self, data_point):
		
		# iterate over the number of blocks
		pr = 0.0
		for i in arange(len(self.comp)):
			pr = pr + log(self.comp[i].prob(data_point[i]))

		return pr

	def getNumberOfBlocks(self):
		return float(len(self.comp))


# choose the list of data sets to run the algorithm on
#dataSetList = ["nltcs", "msnbc", "kdd", "plants", "baudio", "jester", "bnetflix", "msweb", "book", "cwebkb", "cr52", "csmall20ng"]
#dataSetList = ["nltcs", "msnbc", "kdd", "plants"]
#dataSetList = ["baudio", "jester", "bnetflix", "msweb", "book"]
dataSetList = ["cwebkb"]

# iterate over all datasets
for dataSetName in dataSetList:

	print "Starting experiment for data set ",dataSetName,"..."
	print datetime.datetime.now()

	# load the training data
	data = numpy.loadtxt(open(dataSetName+".ts.data","rb"),dtype=int,delimiter=",")

	# get the dimensions of the trainging data matrix
	m,n = data.shape

	# the number of mixture components (latent variable values)
	numComponents = 20

	# generate random matrix
	initData = np.random.randint(2, size=(m, n))

	# split the random matrix. this is used to initialize EM
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

	averageLL = 0.0

	for c in arange(100):
		print "EM iteration: ",c
		# iterate over the training samples (all of them) an compute probability
		compPr = zeros(numComponents, dtype=float)
		weights = zeros((numComponents, m), dtype=float)
		# the E step
		for i in arange(m):
			probSum = 0.0
			for j in arange(numComponents):
				# probability (unnormalized) of the data point i for the component j
				if latentProb[j] > 0.0:
					prob = exp(log(latentProb[j]) + comp[j].probLog(data[i]))
				else:
					prob = 0.0

				weights[j][i] = prob
				# the sum of the probabilites (used for normalization)
				probSum += prob
				#print weights[j][i]
		
			for j in arange(numComponents):
				# normalize the probabilities
				if probSum <= 0.0:
					weights[j][i] = 0.0
				else:
					weights[j][i] = weights[j][i] / probSum
				# aggregate the normalized probabilities
				compPr[j] += weights[j][i]

		# the M step
		# update the class priors
		latentProb = compPr/sum(compPr)

		#print "Difference of probabilities: ",sum(abs(latentProbNew - latentProb))

		# compute the log-likelihood on the training data 
		sumnNew = 0.0
		for x in data:
			prob = 0.0
			for j in arange(numComponents):
				if latentProb[j] > 0.0:
					prob = prob + exp(log(latentProb[j]) + comp[j].probLog(x))

			if prob > 0.0:
				sumnNew += log(prob)
			else:
				print "what the heck?"

		averageLLNew = sumnNew/float(m)

		print "Current average log-likelihood on the training data: ",averageLLNew
		print "Difference in average log-likelihood: ",abs(averageLL - averageLLNew)

		# stop EM when probs are not changing anymore
		if abs(averageLL - averageLLNew) < 0.001:
			break

		# set the old ll to the current one
		averageLL = averageLLNew

		# update the parameters of the mixture components
		# comp are the mixture components
		comp = array([])
		# run inference in the components and compute the new probabilities
		for j in arange(numComponents):
			# create a mixture component for the ith row having value '0'
			compTemp = IndComponent(data, weights[j])
			#print " ---- "
			comp = append(comp, compTemp)
			#print weights[j]


	# load test data
	data_test = numpy.loadtxt(open(dataSetName+".test.data","rb"),dtype=int,delimiter=",")

	# dimensions of test data
	mt,nt = data_test.shape

	# compute the log-likelihood of the test data for the partial exchangeable model
	testCounter = 0
	sumn = 0.0
	for x in data_test:
		prob = 0.0
		for j in arange(numComponents):
			if latentProb[j] > 0.0:
				prob = prob + exp(log(latentProb[j]) + comp[j].probLog(x))
		sumn += log(prob)

	print "Data set: ",dataSetName
	print "Mixture of naive Bayes models: ",sumn / float(mt)


