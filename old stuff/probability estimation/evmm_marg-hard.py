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
	splitNr = 2
	splitNumbers = [0]
	partition = array([])
	smooth = 1.0
	nk = dict()
	sampleWeights = dict()
	laplace = 0.001

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
				sWeight += 1.0
				#print sWeight
				self.sampleWeights[configTuple] = sWeight
			else:
				self.sampleWeights[configTuple] = 1.0


		#print dummyCounter
		# number of configurations that have no occurence in the training data
		diff = prod - len(self.sampleWeights)

		#print "diff: ",diff

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

	# initialize the exchangeable sequence and learn the structure based on the data points
	def __init__(self, data):

				# get the dimensions of the trainging data matrix
		m,n = data.shape

		# compute the priors from the training data: prob(x=1)
		msTemp = zeros(n,dtype=float)
		for i in arange(n):
			msTemp[i] = (float(sum(data[:,i]==1)) + 0.1) / (float(sum(data[:,i]<=1)) + 0.2)

		# represents the assignment to blocks
		assign = zeros(n, dtype=int)
		# counter keeping track of the number of blocks
		countUnique = 0

		# sorted the marginals and keep track of the order
		sortedMsTempArg = argsort(msTemp)
		sortedMsTemp = sort(msTemp)

		# the current value for comparison
		previousValue = sortedMsTemp[0]

		# iterate through the unique marginals
		for j in arange(n):
			# if there is a gap greater than the threshold or a block is larger than 1000 variables
			# then start a new block
			if abs(previousValue-sortedMsTemp[j]) > 0.0001:
					countUnique += 1
					assign[sortedMsTempArg[j]] = countUnique
			else:
				assign[sortedMsTempArg[j]] = countUnique
			previousValue = sortedMsTemp[j]

		# the integers used in part to index the blocks (e.g.: [0, 2, 3])
		self.splitNumbers = unique(assign)
		# copy the partition indicator array to the class variable "partition"
		self.partition = assign
		# initialize array of exchangeable components
		self.comp = array([])

		for i in self.splitNumbers:
			mc,nc = data[:,self.partition==i].shape
			self.comp = append(self.comp, MComponent(data[:,self.partition==i], zeros(nc, dtype=int)))

		print "number of blocks: ",len(list(self.comp))

	# simply update the parameters not the structure
	def update(self, data):

		self.comp = array([])
		for i in self.splitNumbers:
			mc,nc = data[:,self.partition==i].shape
			self.comp = append(self.comp, MComponent(data[:,self.partition==i], zeros(nc, dtype=int)))

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

# here the actual program starts

print datetime.datetime.now()

f_handle = file('result-evmm-cov-all-20-30.log', 'a')
outputString = str(datetime.datetime.now())+" %s"
savetxt(f_handle, array([""]), fmt=outputString)
f_handle.close()

dataSetList = ["nltcs", "msnbc", "kdd", "plants", "baudio", "jester", "bnetflix", "msweb", "book", "webkb", "r52", "small20ng"]
#dataSetList = ["msnbc", "kdd", "plants", "baudio", "jester", "bnetflix", "msweb", "book", "webkb", "r52", "small20ng"]
#dataSetList = ["plants", "baudio", "jester", "bnetflix", "msweb", "book", "webkb", "r52", "small20ng"]

# iterate over all datasets
for dataSetName in dataSetList:

	dataSetName = "book"
	print "Starting experiment for data set ",dataSetName,"..."

	# load the training data
	data = numpy.loadtxt(open(dataSetName+".ts.data","rb"),dtype=int,delimiter=",")

	# load test data
	data_test = numpy.loadtxt(open(dataSetName+".test.data","rb"),dtype=int,delimiter=",")

	# get the dimensions of the trainging data matrix
	m,n = data.shape

	# compute the priors from the training data: prob(x=1)
	ms = zeros(n,dtype=float)
	for i in arange(n):
		ms[i] = mean(data[:,i])

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

	for c in arange(50):
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
		latentProb = compPr/sum(compPr)
		print "latent class probabilities: ",latentProb
		#print "---"

		# update the parameters of the mixture components
		# comp are the mixture components
		comp = array([])

		# run inference in the components and compute the new probabilities
		for j in arange(numComponents):
			# create a mixture component for the ith row having value '0'
			compTemp = IndComponent(data[weights==j,:])
			# append to list of independent components
			comp = append(comp, compTemp)
			# update the "old" mixture components
			#comp[j].update(data[weights==j,:])
			
		# compute the log-likelihood on the training data for both candidate structures
		# and chose the one with the highest log-likelihood score
		#sumn1 = 0.0
		#sumn2 = 0.0
		#for x in data:
		#	prob1 = 0.0
		#	prob2 = 0.0
		#	for j in arange(numComponents):
		#		if latentProb[j] > 0.0:
		#			prob1 = prob1 + latentProb[j] * comp[j].prob(x)
		#			prob2 = prob2 + latentProb[j] * comp2[j].prob(x)
		#	sumn1 = sumn1 + log(prob1)
		#	sumn2 = sumn2 + log(prob2)

		# if the new structure is better, use this one
		#if sumn2 >= sumn1:
		#	comp = comp2		

		# compute the log-likelihood of the test data for the partial exchangeable model
		sumn = 0.0
		for x in data_test:
			prob = 0.0
			for j in arange(numComponents):
				if latentProb[j] > 0.0:
					prob = prob + latentProb[j] * comp[j].prob(x)
			sumn = sumn + log(prob)

		print "Mixture of independent EVMs: ",sumn / len(data_test)


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


