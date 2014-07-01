import numpy as np
from operator import mul
import operator
import scipy
from collections import Counter
from numpy import random
import sys, traceback
import datetime
from scipy.cluster.vq import kmeans2

# computes the binomial coefficient
def  n_take_k(n,r):
  
    if r > n-r:  # for smaller intermediate values
        r = n-r
    return int( reduce( mul, range((n-r+1), n+1), 1) /
      reduce( mul, range(1,r+1), 1) )


# class representing one mixture component
class MComponent:
	size = 1	
	smooth = 1.0
	nk = dict()
	sampleWeights = dict()
	laplace = 0.1

	# part is the partition in form of an array [0, 0, 1, 0, 1, 2, 3, 1] indicating column membership on (here: 4) partitions
	def __init__(self, data):
		
		# get dimensions of the data matrix
		m,n = data.shape

		# stores the possible binomial coefficients (caching)
		self.nk = dict()
		# stores the accumulated weights 
		self.sampleWeights = dict()
		# size of the block
		self.size = n

		# go through all rows of the data table and store number of configurations
		for i in arange(m):
			configs = count_nonzero(data[i])
			
			sWeight = self.sampleWeights.get(configs, -1.0)
			#print sWeight
			if sWeight > 0:
				sWeight += 1.0
				#print sWeight
				self.sampleWeights[configs] = sWeight
			else:
				self.sampleWeights[configs] = 1.0

		# number of configurations that have no occurence in the training data
		diff = n - len(self.sampleWeights) + 1

		#  the normalization constant for the probabilities of the block configurations
		self.smooth = float(sum(self.sampleWeights.values()))

		#print self.smooth

		# perform Laplacian smoothing -> add delta to each possible configuration
		self.smooth += diff*self.laplace
		for i in list(self.sampleWeights):
			self.sampleWeights[i] += self.laplace
			self.smooth += self.laplace

		#print self.sampleWeights

	# returns the probability of one particular configuration (here: conditional probability)
	def prob(self, data_point):

		# iterate over the number of blocks
		configs_test = count_nonzero(data_point)

		# look up the probability of the given block configuration
		sw = self.sampleWeights.get(configs_test, -1.0)		
		if  sw > 0.0:
			currProb = float(sw)/self.smooth
			#print "No SMOOTH! -- ",currProb
		else:
			# if the configuration probability is zero and we are fully exchangeable, apply smoothing
			currProb = self.laplace/self.smooth

		# normalize by the number of configuration represented by this particular block configuration
		nvalue = self.size
		kvalue = configs_test
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

	# initialize the exchangeable sequence and learn the structure based on the data points
	def __init__(self, data):

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
			self.comp = append(self.comp, MComponent(data[:,self.partition==i]))

		#print "number of blocks: ",len(list(self.comp))

	# simply update the parameters not the structure
	def update(self, data):

		self.comp = array([])
		for i in self.splitNumbers:
			mc,nc = data[:,self.partition==i].shape
			self.comp = append(self.comp, MComponent(data[:,self.partition==i]))

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

# here the actual program starts

print datetime.datetime.now()

f_handle = file('result-evmm-cov-all-20-30.log', 'a')
outputString = str(datetime.datetime.now())+" %s"
savetxt(f_handle, array([""]), fmt=outputString)
f_handle.close()

#dataSetList = [1]
#dataSetList = ["nltcs", "msnbc", "kdd", "plants", "baudio", "jester", "bnetflix", "msweb", "book", "webkb", "r52", "small20ng"]
#dataSetList = ["baudio", "jester", "bnetflix", "msweb", "book", "r52", "small20ng"]
dataSetList = ["small20ng", "r52", "book"]
#dataSetList = ["msnbc", "kdd", "plants", "baudio", "jester", "bnetflix", "msweb", "book", "webkb", "r52", "small20ng"]
#dataSetList = ["plants", "baudio", "jester", "bnetflix", "msweb", "book", "webkb", "r52", "small20ng"]

# iterate over all datasets
for dataSetName in dataSetList:

	#dataSetName = "webkb"
	print "Starting experiment for data set ",dataSetName,"..."
	print datetime.datetime.now()

	# load the training data
	data = numpy.loadtxt(open(dataSetName+".ts.data","rb"),dtype=int,delimiter=",")

	# load test data
	data_test = numpy.loadtxt(open(dataSetName+".test.data","rb"),dtype=int,delimiter=",")

	# get the dimensions of the trainging data matrix
	m,n = data.shape

	# the number of mixture components (latent variable values)
	numComponents = 20

	# generate random matrix
	initData = np.random.randint(2, size=(m, n))
	
	# split the random matrix. this is used to initialize EM
	initSplit = array_split(initData, numComponents, axis=0)

	# comp are the mixture components
	comp = array([])
	for j in arange(numComponents):
		# create a mixture component for the ith row having value '0'
		compTemp = IndComponent(initSplit[j])
		#compTemp = IndComponent(array([initData[j,:]]))
		# the array with the mixture components
		comp = append(comp, compTemp)

	# class probabilities initialized to uniform probabilities
	latentProb = ones(numComponents, dtype=float)
	latentProb = latentProb/sum(latentProb)

	print "latent class probability: ",latentProb

	for c in arange(100):
		print "EM iteration: ",c
		# iterate over the training samples (all of them) an compute probability
		compPr = zeros(numComponents, dtype=float)
		weights = zeros(m, dtype=float)
		# the E step
		for i in arange(m):
			probSum = 0.0
			pr = zeros(numComponents, dtype=float)
			for j in arange(numComponents):
				# probability (unnormalized) of the data point i for the component j
				if latentProb[j] > 0.0:
					pr[j] = log(latentProb[j]) + comp[j].probLog(data[i])
				else:
					pr[j] = -inf
		
			#print " "
			#print weights
			#print probSum
			maxProb = argmax(pr)

			weights[i] = maxProb
			# aggregate the normalized probabilities
			compPr[maxProb] += 1.0
			

		# the M step
		# update the class priors
		latentProbNew = compPr/sum(compPr)

		print "Difference of probabilities: ",sum(abs(latentProbNew - latentProb))

		# stop EM when probs are not changing anymore
		if sum(abs(latentProbNew - latentProb)) < 0.001:
			break

		latentProb = latentProbNew

		#print "latent class probabilities: ",latentProb
		#print "---"

		# update the parameters of the mixture components
		# comp are the mixture components
		comp = array([])

		blockStatistics = zeros(numComponents, dtype=float)

		# run inference in the components and compute the new probabilities
		for j in arange(numComponents):
			# create a mixture component for the ith row having value '0'
			compTemp = IndComponent(data[weights==j,:])
			# append to list of independent components
			comp = append(comp, compTemp)
			# update the "old" mixture components
			#comp[j].update(data[weights==j,:])
			blockStatistics[j] = compTemp.getNumberOfBlocks()
			
		print "blocks;  mean: ", mean(blockStatistics), "; stddev: ", std(blockStatistics)
		

	# compute the log-likelihood of the test data for the partial exchangeable model
	sumn = 0.0
	for x in data_test:
		prob = 0.0
		for j in arange(numComponents):
			if latentProb[j] > 0.0:
				prob = prob + latentProb[j] * comp[j].prob(x)
		sumn = sumn + log(prob)

	print "Mixture of independent EVMs: "+dataSetName+" (final result): ",sumn / len(data_test)

	# print result to file
	outputString = "Mixture of independent EVMs: "+dataSetName+" (final result): %f"
	f_handle = file('rresult-evmm-cov-all-20-30.log', 'a')
	savetxt(f_handle, array([float(sumn / len(data_test))]), fmt=outputString)
	outputString = str(datetime.datetime.now())+" %s"
	savetxt(f_handle, array([""]), fmt=outputString)
	f_handle.close()


