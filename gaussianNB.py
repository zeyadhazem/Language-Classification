from classifier import Classifier
import pandas as pd
import math


class GaussianNBClassifier(Classifier):
    def __init__(self):
        """
        You should pass in any arguments that you think will need for your classifier
        """
        #         Save model for testing
        #         self.model = None
        self.mean = None
        self.std = None
        self.X_train = None
        self.y_train = None
        self.classes = None
        return

    # fit will summarize mean and stdev of each feature for each class into a dictonary called fitdata
    def fit(self, X, y):
        """
        X: A pandas dataframe containing the features extracted
        y: A pandas dataframe that contains the categories of each entry of X

        Implement how you are going to be fitting your model to the data you are presented with
        """

		#Add Y as a new column named Category
        X.loc[:,'Category'] = pd.Series(y, index=X.index)
        self.X_train = X

        # Separate data by class
        self.classes = self.X_train.groupby('Category')

        # Calculate Mean of each feature for each class
        self.mean = self.classes.mean()
        #print self.mean

        # Calculate standard deviation of each feature for each class
        self.std = self.classes.std()
        #print self.std

    # Calculates the gaussian distribution (probability) given a point, the mean and standard deviation
    # We are assuming a normal distribution for the probabilities, hence the gaussian
    def gaussianPdf(self, x, mean, std):
        exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exp

    def calculateProbability(self, data_row):
        #         mean_features, std_features = fit(self.X,self.y)
        Py = [0, 0, 0, 0, 0]

        for i in range(5):
            Py[i] = float(len(self.classes.get_group(i))) / float(len(self.X_train))

        probabilities = [0, 0, 0, 0, 0]
        for i in range(5):
            prob = 1
            for j in range(len(data_row)):
                mean = self.mean.iloc[i, j]
                std = self.std.iloc[i, j]
                if std != 0.:
                    prob *= self.gaussianPdf(data_row[j], mean, std)
            probabilities[i] = prob * Py[i]
        return probabilities

    def predict(self, test_set):
        """
        return an array that is the same size of test_set and has the classified categories
        """
		
        predictions = []
        all_prob = []
        for i in range(test_set.shape[0]):
            print i,
            p = self.calculateProbability(test_set.iloc[i,:])
            all_prob.extend([p])
            best_class = all_prob[i].index(max(p))
            predictions.append(best_class)
        return predictions
