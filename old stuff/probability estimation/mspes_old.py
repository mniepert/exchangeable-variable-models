import numpy
import scipy
from collections import Counter
from collections import deque
from scipy.cluster.vq import vq, kmeans2, whiten

def dequefilter(deck, condition):
	for _ in xrange(len(deck)):
		item = deck.popleft()
		if condition(item):
			deck.append(item)

# class representing one mixture component
class MComponent:
	splitNr = 2
	peSplit = [1]
	diff = 1
	c = Counter([])
	smooth = 1.0

	def __init__(self, data, peSplit):
		
		self.peSplit = peSplit
		self.splitNr = len(peSplit)+1
		proj = array_split(data, self.peSplit, axis=1)
		# get dimensions of the components
		mx,nx = proj[0].shape

		# get dimensions of the data matrix
		m,n = data.shape

		#build a dictionary that stores the probs
		configs = zeros(m, dtype=str(self.splitNr)+'int')

		#the number of configurations
		prod = 1.0
		for i in arange(self.splitNr):
			prod = prod * proj[i].shape[1]

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

		self.smooth = float(m)+float(self.diff)

	def prob(self, data_point):

		# here we need to convert a vector to the condensed representation
		proj_data_point = split(data_point, self.peSplit, axis=1)

		configs_test = zeros(self.splitNr, dtype=int)

		for i in arange(self.splitNr):
			configs_test[i] = count_nonzero(proj_data_point[i])

		x = tuple(configs_test)

		if self.c[x] > 0:
			return float(self.c[x])/self.smooth
		else:
			return 1.0/self.smooth


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
		
		

data = numpy.loadtxt(open("plants.ts.data","rb"),dtype=float,delimiter=",")

m,n = data.shape
ms = zeros(n,dtype=float)
for i in arange(n):
	ms[i] = average(data[:,i])

A = argsort(ms)
ms = sort(ms)

# sort the array according to marginals in ascending order
data = data[arange(data.shape[0])[:,newaxis],A]

#centers,assign = kmeans2(ms, 10, 100)

#print assign
#print data[:,assign==8]



##############################################################
# here we use the validation data to learn the mixture weights

# load test data
data_valid = numpy.loadtxt(open("plants.valid.data","rb"),dtype=float,delimiter=",")

# sort the array according to marginals in ascending order
data_valid = data_valid[arange(data_valid.shape[0])[:,newaxis],A]

numMixtures = 5

wsum = 0.0
comp = array([])
w = array([])
pw = array([])
for i in arange(numMixtures):
	sep1 = (i+2)*10
	ar = array([10, sep1])
	celem = MComponent(data, ar)
	comp = append(comp, celem)
	w = append(w, comp[i].ll(data_valid))
	wsum = wsum + w[i]

for i in arange(numMixtures):
	pw = append(pw, w[i] / wsum)


print pw

############################################################################
# the evaluation starts

# load test data
data_test = numpy.loadtxt(open("plants.test.data","rb"),dtype=float,delimiter=",")

mt,nt = data_test.shape

# sort the array according to marginals in ascending order
data_test = data_test[arange(data_test.shape[0])[:,newaxis],A]

# compute the log-likelihood of the model that assumes independence between all variables
# go through the rows of the test data matrix and compute the log-likelihood of the test data
sum_ind = 0.0
counter = 0
for i in arange(mt):
	pr = 1.0
	for j in arange(nt):
		pr = pr*((1.0-data_test[i][j])*(1.0-ms[j])+data_test[i][j]*ms[j])
	if pr > 0:
		sum_ind = sum_ind + log(pr)
	else:
		counter += 1
		sum_ind = sum_ind + log(1.0/m)

print "Independent model: ",sum_ind / mt

# convert array to set of tuples
configs_test_hash = map(tuple, data_test)

# compute the log-likelihood of the test data for the partial exchangeable model
sumn = 0.0
for x in configs_test_hash:
	prob = 0.0
	for i in arange(numMixtures):
		prob = prob + pw[i] * comp[i].prob(x)

	#print prob
	sumn = sumn + log(prob)

print sumn / mt

#print comp.ll(data_test)

