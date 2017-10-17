from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd

from preprocessor import Preprocessor

class Training:
    def __init__(self, X, y, feature_extraction_pipeline, model):
        self.X = X
        self.y = y
        self.feature_extraction_pipeline = feature_extraction_pipeline
        self.model = model

    def train (self, validation, num_features=None):
        if validation:
            # split training set into train and validate
            print("> Separating training set into train and validation")
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)

        else:
            X_train = self.X
            y_train = self.y

            # Read test set
            X_test = pd.read_csv('test_set_x.csv')
            X_test.drop('Id', axis=1, inplace=True)
            Preprocessor().process(X_test, inplace=True)

            # Apply processing on test set
            print("> Extracting features from test set")
            for feature_extractor in self.feature_extraction_pipeline:
                applied_features = feature_extractor.applyToTest(X_test)
                X_test = pd.concat([X_test, feature_extractor.addPrefix(applied_features)], axis=1)

            # No need for the text column anymore
            X_test = X_test.drop('Text', axis=1)

        if num_features:
            featureSelection = SelectKBest(chi2, k=num_features)
            X_train = pd.DataFrame(featureSelection.fit_transform(X_train, y_train))
            X_test = pd.DataFrame(featureSelection.transform(X_test))
            print("Feature Selection: X_train:", X_train.shape, "X_test:", X_test.shape)

        # Train model
        print("> Training the model")
        self.model.fit(X_train, y_train.values.ravel())

        print("> Predicting")
        prediction = self.model.predict(X_test)

        if validation:
            print(confusion_matrix(y_test, prediction))
            print('\n')
            print(classification_report(y_test, prediction))
            print(accuracy_score(y_test, prediction))

        else:
            print("> Exporting")
            results = pd.DataFrame({'Category':prediction})
            results.index.names = ['Id']

            results.to_csv("results.csv")
