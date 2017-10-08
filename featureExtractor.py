class FeatureExtractor:
    def __init__(self, train_set, prefix):
        self.train_set = train_set
        self.feature_prefix = prefix

    def extractFeatures(self):
        return

    def applyToTest(self, test_set):
        return test_set

    def addPrefix(self, df):
        columns = df.columns
        new_columns = []
        for column in columns:
            new_columns.append(self.feature_prefix + "" + column)

        df.columns = new_columns
        return df
