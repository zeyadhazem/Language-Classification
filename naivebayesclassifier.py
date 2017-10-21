from classifier import Classifier
import pandas as pd
import math
from collections import Counter

class NaiveBayesClassifier(Classifier):
	def __init__(self):
		"""
		You should pass in any arguments that you think will need for your classifier
		"""
		self.prob_given = {}
		self.prob_of = {}

	def fit(self, X, y):
		"""
		X: A dataframe containing the features extracted
		y: A dataframe that contains the categories of each entry of X

		Implement how you are going to be fitting your model to the data you are presented with
        """

		#Merge with class so that we can group
		X.loc[:,'Category'] = pd.Series(y, index=X.index)
		grouped = X.groupby('Category')

		# build P(class)
		self.prob_of = grouped.size().apply(lambda y : float(y) / len(X.index)).to_dict()

		# build P(letter|class)
		for lang, group in grouped:
			group.drop("Category", axis=1, inplace=True)
			print("Language:",lang, "Prob:",self.prob_of[lang])
			print(type(group), group.shape)
			self.prob_given[lang] = {}
			for feature in group.columns.values:
				self.prob_given[lang][feature] = group[feature].mean()

		return
	
	def predict(self, test_set):
		"""
		return an array that is the same size of test_set and has the classified categories
		"""
		prediction = []
		
		categories = self.prob_of.keys()

		#planning to predict like so : P(lang)*P(letter|lang) = P(lang) *  P(a|lang)*tf(a)  *  P(b|lang)*tf(b)  * ...  *  P(z|lang)*tf(z) * ... more tfidf
		for i, series in test_set.iterrows():
			#print "TEST_SET ROW",i
			w = series[series != 0].prod() #tf(a) * tf(b) * ... tf(b)
			#print -math.log10(series[series != 0].prod()) #smalles num is bigggg
			p = { k:1.0 for k in categories }
			
			#calculate likelihood
			for feature, weight in series[series != 0].iteritems():
				for lang in categories:
					p[lang] = p[lang] * self.prob_given[lang][feature] 

			#add the weight
			for lang in categories:
				p[lang] = p[lang] * w * self.prob_of[lang]

			#find the biggest prob
			sorted_p = sorted(p.iteritems(), key=lambda(k,v):(v,k), reverse=True)
			prediction.append(sorted_p[0][0])

		return prediction
