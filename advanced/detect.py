import numpy as np
from operator import mul
import operator
import scipy
from collections import Counter
from collections import deque
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from decimal import *

# given a permutation (e.g. [1 3 2 4 5 6 7]) apply it to a matrix
def permutateIndexes(array, perm):
    return array[np.ix_(*(perm[:s] for s in array.shape))]


dataSetName = "tmovie"

# load the training data
data = loadtxt(open(dataSetName+".ts.data","rb"),dtype=int,delimiter=",")

# get the dimensions of the trainging data matrix
m,n = data.shape

# compute the marginal probabilities of the involved variables
ms = zeros(n,dtype=float)
for i in arange(n):
	ms[i] = average(data[:,i])


#A = argsort(ms)
#ms = sort(ms)

# sort the array according to marginals in ascending order
#data = data[arange(data.shape[0])[:,newaxis],A]


dmatrix = zeros((n, n))

# convert the 2d array into a set of tuples
cdata = map(tuple, data)

c = Counter(cdata)

for i in arange(n):
	for j in arange(i+1, n):
		#apply permutation to matrix
		#pData = permutateIndexes(data, permutation)
		pData = copy(data)
		pData[:,[i, j]] = pData[:,[j, i]]
		tpData = map(tuple, pData)
		
		# count number of configurations
		pc = Counter(tpData)
		# compute difference (= proxy for variation distance)
		result = c - pc

		#print pc.most_common(1)

		ndiff = sum(result.values())/float(m)
		dmatrix[i][j] = ndiff
		dmatrix[j][i] = ndiff

		print i,"  ",j,"  ; difference: ",sum(result.values())/float(m)


savetxt(dataSetName+"_matrix.out", dmatrix, delimiter=',')




