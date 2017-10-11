from classifier import Classifier

class NaiveBayesClassifier(Classifier):
    def __init__(self):
        """
        You should pass in any arguments that you think will need for your classifier
        """
        return

    def fit(self, X, y):
        """
        X: A dataframe containing the features extracted
        y: A dataframe that contains the categories of each entry of X

        Implement how you are going to be fitting your model to the data you are presented with
        """
        return

    def predict(self, test_set):
        """
        return an array that is the same size of test_set and has the classified categories
        """
        return [0] * len(test_set)
