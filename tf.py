import pandas as pd
from featureExtractor import FeatureExtractor

class TF (FeatureExtractor):
    """Tokenizes the input and calculates the frequencies"""

    def __init__(self, df):
        """Initialize a TF instance"""
        FeatureExtractor.__init__(self, df, 'tf_') # A vector of size m that will be tokenized
        self.tokens = {} # A dictionary of unique tokens where each has a list of size m with the token count by training example
        # Make sure size = m * 1

    def extractFeatures(self):
        rows_len = len(self.train_set.index)
        column = self.train_set.columns[0]

        for i in range (0, rows_len):
            string = self.train_set[column][i].replace(" ","")
            string_len = len(string)

            for letter in string:
                lower = letter.lower()
                if lower not in self.tokens.keys():
                    self.tokens[lower] = [0] * rows_len

                self.tokens[lower][i] = self.tokens[lower][i] + 1.0/string_len

        return_df = pd.DataFrame(self.tokens)

        return return_df

    def applyToTest(self, test_df):
        """
        test_df: m * 1 array
        """
        test_df_len = len(test_df.index)
        test_tokens = {}

        # Initialize an empty tokens dict
        for token in self.tokens.keys():
            test_tokens[token] = [0] * test_df_len

        # Get unique chars with the same dimensions as in the test set
        column = test_df.columns[0]
        for i in range (0, test_df_len):
            string = test_df[column][i].replace(" ","")
            string_len = len(string)

            for letter in string:
                lower = letter.lower()
                if lower in test_tokens.keys():
                    test_tokens[lower][i] = test_tokens[lower][i] + 1.0/string_len

        return_df = pd.DataFrame(test_tokens)

        return return_df
