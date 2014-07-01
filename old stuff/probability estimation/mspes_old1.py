import numpy as np
import scipy
from collections import Counter
from collections import deque
from scipy.cluster.vq import vq, kmeans2, whiten

def dequefilter(deck, condition):
	for _ in xrange(len(deck)):
		item = deck.popleft()
		if condition(item):
			deck.append(item)

def n_take_k(n,k):
        """Returns (n take k),
        the binomial coefficient.

        :since: 2005-11-17
        """
        n, k = int(n), int(k)
        assert (0<=k<=n), "n=%f, k=%f"%(n,k)
        k = min(k,n-k)
        c = 1
        if k>0:
                for i in xrange(k):
                        c *= n-i
                        c //= i+1
        return c

# class representing one mixture component
class MComponent:
	splitNr = 2
	peSplit = [1]
	blockSize = array([])
	diff = 1
	c = Counter([])
	smooth = 1.0

	def __init__(self, data, peSplit, exch):
		
		# get dimensions of the data matrix
		m,n = data.shape

		if exch:
			# full exchangeability
			#build a dictionary that stores the probs
			configs = zeros((m,1))

			# go through all rows of the data table and store number of configurations
			for i in arange(m):
				configs[i][0] = count_nonzero(data[i,:])

			prod = n
		else:
			self.peSplit = peSplit
			self.splitNr = len(peSplit)+1
			proj = array_split(data, self.peSplit, axis=1)
			
			#build a dictionary that stores the probs
			configs = zeros(m, dtype=str(self.splitNr)+'int')

			#the number of configurations
			prod = 1.0
			self.blockSize = zeros((self.splitNr,),dtype=int)
			for i in arange(self.splitNr):
				prod = prod * proj[i].shape[1]
				self.blockSize[i] = proj[i].shape[1]


			# go through all rows of the data table and store number of configurations
			for i in arange(m):
				for j in arange(self.splitNr):
					configs[i][j] = count_nonzero(proj[j][i,:])

		# only take those elements with a count > 1
		configs = map(tuple, configs)
		deck = deque(configs)
		dequefilter(deck, lambda x: x > 1)

		# map the array of all configurations to a hash table of tuples
		configs_hash = map(tuple, deck)

		# count the frequencies of the different configurations
		self.c = Counter(configs_hash)

		# number of configurations that have 1 or no occurence
		self.diff = prod - len(self.c)

		#print "m: ",m
		#print "prod: ",prod
		#print "sum: ",sum(self.c.values())
		#print self.c.most_common(10)

		self.smooth = float(m)+float(self.diff)

	def prob(self, data_point):

		# here we need to convert a vector to the condensed representation
		proj_data_point = split(data_point, self.peSplit, axis=1)

		configs_test = zeros(self.splitNr, dtype=int)

		normC = 1.0
		for i in arange(self.splitNr):
			configs_test[i] = count_nonzero(proj_data_point[i])
			normC = normC * n_take_k(self.blockSize[i],configs_test[i])

		x = tuple(configs_test)

		if self.c[x] > 1:
			return (float(self.c[x])/self.smooth)/normC
		else:
			return (1.0/self.smooth)/normC


	def ll(self, data):
		# get dimensions of data matrix
		m,n = data.shape

		# convert array to set of tuples
		data_hash = map(tuple, data)

		# compute the log-likelihood of the test data for the partial exchangeable model
		sumn = 0.0
		for x in data_hash:
			sumn = sumn + log(self.prob(x))

		return sumn / m

	def llt(self, data):
		# get dimensions of data matrix
		m,n = data.shape

		# convert array to set of tuples
		data_hash = map(tuple, data)

		# compute the log-likelihood of the test data for the partial exchangeable model
		sumn = 0.0
		for x in data_hash:
			sumn = sumn + log(self.prob(x))

		return sumn
		
		

data = numpy.loadtxt(open("plants.ts.data","rb"),dtype=float,delimiter=",")

m,n = data.shape
ms = zeros(n,dtype=float)
for i in arange(n):
	ms[i] = average(data[:,i])

#p = permutation(len(ms))

#print p

#ms = ms[p]

#A = argsort(ms)
#ms = sort(ms)
#print ms

# sort the array according to marginals in ascending order
#data = data[arange(data.shape[0])[:,newaxis],A]
#data = data[:,p]

centers,assign = kmeans2(ms, 10, 100)

#print assign
#print data[:,assign==8]



##############################################################
# here we use the validation data to learn the mixture weights
##### HARD EM ################################################

# load test data
data_valid = numpy.loadtxt(open("plants.valid.data","rb"),dtype=float,delimiter=",")

# get dimensions of validation data matrix
mv,nv = data_valid.shape

# sort the array according to marginals in ascending order
data_valid = data_valid[arange(data_valid.shape[0])[:,newaxis],A]
#data_valid = data_valid[:,p]

# the number of mixture components
numMixtures = 1

comp = array([])
for i in arange(numMixtures):
	ar = array([1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 24, 28, 32, 34, 36, 40, 44, 47, 49, 54, 57, 59, 65, 66, 67, 68])
	celem = MComponent(data, ar, False)
	comp = append(comp, celem)

# convert array to set of tuples
data_valid_hash = map(tuple, data_valid)

maxCounter = zeros(numMixtures)

# compute the log-likelihood of the test data for the partial exchangeable model
for x in data_valid_hash:
	maxProb = 0.0
	maxComp = 1
	for i in arange(numMixtures):
		tprob = comp[i].prob(x)
		if tprob > maxProb:
			maxProb = tprob
			maxComp = i
	maxCounter[maxComp] += 1

#print maxCounter
pw = map(float, maxCounter/mv)
print "Mixture weights: ",pw
#print "tCounter: ",tCounter
#print "validationmatrixSize: ",mv


############################################################################
# the evaluation starts

# load test data
data_test = numpy.loadtxt(open("plants.test.data","rb"),dtype=float,delimiter=",")

mt,nt = data_test.shape

# sort the array according to marginals in ascending order
data_test = data_test[arange(data_test.shape[0])[:,newaxis],A]
#data_test = data_test[:,p]

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
		sum_ind = sum_ind + log(1.0/m)

print testCounter
print "Independent model: ",sum_ind / mt

# convert array to set of tuples
configs_test_hash = map(tuple, data_test)

# compute the log-likelihood of the test data for the partial exchangeable model
testCounter = 0
sumn = 0.0
for x in configs_test_hash:
	testCounter += 1
	prob = 0.0
	for i in arange(numMixtures):
		prob = prob + pw[i] * comp[i].prob(x)

	#print prob
	sumn = sumn + log(prob)

print testCounter
print "Mixture of smoothed partially exchangeable sequences: ",sumn / mt

#print comp.ll(data_test)

