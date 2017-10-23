from classifier import Classifier
from collections import Counter
from sklearn.neighbors import KDTree
import numpy as np
import math

class KNearestNeighbor(Classifier):

	def __init__(self, K=None, wfunc=None, optimization=None):
		print "> Using K Nearest Neighbor Classifier"
		self.X_train = {}
		self.Y_train = []
		self.K = K
		self.wfunc = wfunc
		self.optimization = optimization
		self.tree = None

	def fit(self, X, y):
		"""
		X: A dataframe containing the features extracted 
		y: A dataframe containing the categories of each entry x
		"""

		self.X_train = X.reset_index(drop=True)
		self.Y_train = y

		if self.optimization == 'kdtree':
			self.tree = KDTree(X)

		return

	def predict(self, test_set):
		prediction = []

		# using distance-weigther or neighbors based on input parameters
		if self.K:
			def getPrediction(test_row):
				return self.getNearestNeighborsPrediction(self.X_train, self.Y_train, test_row, self.K)
		elif self.wfunc:
			def getPrediction(test_row):
				return self.getDWNeighborsPrediction(self.X_train, self.Y_train, test_row, self.wfunc)

		for row in range(len(test_set.index)):
			print row,
			c,p = getPrediction(test_set.iloc[row])
			prediction.append(c)

		print ""

		return prediction

	def getDistanceWeight(self, data_x, new_x, wfunc):
		""" Takes 2 pandas.Series objects: data_x and new_x; weight function: wfunc """
		distance = self.getEuclideanDistance(data_x, new_x)
		return wfunc( distance )

	def getDWNeighborsPrediction(self, X_train, Y_train, new_x, wfunc):
		
		weight_sum = 0
		prediction = 0

		for row, series in X_train.iterrows():
			w = self.getDistanceWeight(series, new_x, wfunc)

			weight_sum += w
			y = Y_train[row]+1 #shift by 1 because classes are from [0,4]

			prediction += w*y

		prediction = (prediction / weight_sum) - 1 #shift back 1
		class_value = int(round(prediction))
		return (class_value, prediction)

	def getEuclideanDistance(self, data_x, new_x):
		""" Takes 2 pandas.Series objects: data_x and new_x """
		#for p, q in zip(data_x, new_x):
		#	d += (p - q)**2

		#return math.sqrt(d)
		return  np.sqrt(np.sum(np.square(data_x - new_x))) #optimization

	def getNearestNeighborsPrediction(self, X_train, Y_train, new_x, k):
		""" Takes pandas.DataFrame: X_train; numpy.ndarry: Y_train, pandas.Series new_x; and int: k"""
		neighbors = []

		if self.optimization == 'kdtree':
			distances, neighbors = self.tree.query([new_x], k=k)
			neighbors = [Y_train[n] for n in neighbors[0]]
		else:
			d = {}

			for i in range(len(X_train.index)):
				d[i] = self.getEuclideanDistance(X_train.iloc[i], new_x)

			sorted_d = sorted(d.iteritems(), key = lambda (k,v) : (v,k))


			for i in range(k):
				row, distance = sorted_d[i]
				neighbors.append(Y_train[row])

		#print neighbors

		c = Counter(neighbors)
		class_value = c.most_common(1)[0][0]
		likelihood = c.most_common(1)[0][1] / float(len(neighbors))

		return (class_value, likelihood)

