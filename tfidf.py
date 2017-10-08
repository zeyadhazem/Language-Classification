import pandas as pd
import math
from featureExtractor import FeatureExtractor
from tf import TF

class TFIDF (FeatureExtractor):
    """Feature Extraction using TFIDF"""

    def __init__(self, train_set, category_df):
        """Initialize a TFIDF instance"""
        FeatureExtractor.__init__(self, train_set, 'tfidf_') # An m * 1 DataFrame that contains the strings whose letter frequencies are to be calculated
        self.category_df = category_df # A vector of size m that contains the category
        self.idf = None
        self.tf = TF(self.train_set)
        self.tf_df = self.tf.extractFeatures() # Get the term frequencies

    def extractFeatures(self):
        idf = self.get_idf()
        return_df = self.tf_df.mul(idf, axis=1)

        return return_df

    def get_idf(self):
        if self.idf is None:
            # Concat the 2 DataFrames together
            df = pd.concat([self.tf_df, self.category_df], axis=1)

            # Group the data by category
            category = self.category_df.columns[0]
            group = df.groupby(category).sum().transpose()

            # Get the char doc frequency
            char_doc_freq_df = group.astype(bool).sum(axis=1)

            # Apply the inverse document frequency equation
            num_categories = len(group.columns)
            self.idf = char_doc_freq_df.apply(lambda x:math.log10(num_categories/x))

        return self.idf

    def applyToTest(self, test_df):
        tf_df = self.tf.applyToTest(test_df)
        return tf_df.mul(self.get_idf(), axis=1)
