import random
import string as strlib
import itertools
import pandas as pd
from collections import Counter
from featureExtractor import FeatureExtractor

class NGRAMS(FeatureExtractor):
	""" Features extraction using n-grams """

	def __init__(self,df):
		""" Initializes a NGRAMS instance """
		FeatureExtractor.__init__(self, df, 'ngrams_')
		self.tokens = {} # A dictionary of unique and unordered ngrams where each has a list of size m examples of counts

	def generateNLetterGrams(self, chars, size, combinations):
		""" Using the 26 letters of alphabet which are common to all 5 languages """

		if not combinations:
			combinations = chars

		if size == 1:
			return combinations

		n_grams = []
		for each in combinations:
			for char in chars:
				n_grams.append(each + char)

		return self.generateNLetterGrams(chars, size-1, n_grams)


	def computeNGramFrequencyAt(self, i, string, n_gram):
		""" Computes the ngram frequency at each training example """
		ngram_count = reduce(lambda x,y : x+y, [ len(word)-1 for word in string])

		if ngram_count != 0:
			for word in string:
				if len(word) >= len(n_gram[0]):
					for x, y in itertools.izip(word, word[1:]): 
						elem = x+y
						if elem in self.tokens.keys():
							self.tokens[elem][i] += 1.0/ngram_count

	#Need to make sure that data has space -> otherwise will be extracting noise
	def extractFeatures(self):
		rows_len = len(self.train_set.index)
		column = self.train_set.columns[0]

		_2gram = self.generateNLetterGrams(list(strlib.ascii_lowercase),2,[])

		self.tokens = {el:[0]*rows_len for el in _2gram}

		for i in range(0, rows_len):
			string = self.train_set[column][i]
			self.computeNGramFrequencyAt(i,string.split(" "), _2gram)

		#collapse unordered ngrams
		extracted_tokens = {}
		for ngram in self.tokens.iterkeys():
			sorted_ngram = ''.join(sorted(ngram))
			if sorted_ngram in extracted_tokens:
				combined_ngram = [extracted_tokens[sorted_ngram], self.tokens[ngram]]
				extracted_tokens[sorted_ngram] = [ sum(x) for x in zip(*combined_ngram) ] 
			else:
				extracted_tokens[sorted_ngram] = self.tokens[ngram]

		self.tokens = extracted_tokens
		return_df = pd.DataFrame(self.tokens)

		return return_df

	def applyToTest(self, test_df):
		test_df_len = len (test_df.index)
		test_tokens = {}
		
		#initialize
		for token in self.tokens.keys():
			test_tokens[token] = [0] * test_df_len

		column = test_df.columns[0]
		for i in range(0, test_df_len):
			string = test_df[column][i].replace(" ", "")
			chars = dict(Counter(string))

			sorted_chars = sorted(filter(lambda x : ord(x) < 128, chars.keys()))

			ngram_occurence = {}
			ngram_len = 0
			for x in sorted_chars:
				for y in sorted_chars:
					elem = x + y
					if elem in self.tokens: 
						num = min(chars[x], chars[y])
						ngram_occurence[elem] = num
						ngram_len += num
			
			for ngram, count in ngram_occurence.items(): 
				test_tokens[ngram][i] = float(count) / ngram_len

		return_df = pd.DataFrame(test_tokens)

		return return_df
			
			
