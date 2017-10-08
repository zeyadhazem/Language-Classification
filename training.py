from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from preprocessor import Preprocessor
import pandas as pd

class Training:
    def __init__(self, X, y, feature_extraction_pipeline):
        self.X = X
        self.y = y
        self.feature_extraction_pipeline = feature_extraction_pipeline

    def train (self, validation, export):
        print("> Getting the features from test set")

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
            print("> Extracting Features from test set")
            for feature_extractor in self.feature_extraction_pipeline:
                applied_features = feature_extractor.applyToTest(X_test)
                X_test = pd.concat([X_test, feature_extractor.addPrefix(applied_features)], axis=1)

            # No need for the text column anymore
            X_test = X_test.drop('Text', axis=1)

        # Train model
        print("> Training the model")
        rfc = RandomForestClassifier(n_estimators=400)
        rfc.fit(X_train, y_train.values.ravel())
        rfc_pred = rfc.predict(X_test)

        if validation:
            print(confusion_matrix(y_test, rfc_pred))
            print('\n')
            print(classification_report(y_test, rfc_pred))
            print(accuracy_score(y_test, rfc_pred))


        if export:
            print("> Exporting")
            results = pd.DataFrame({'Category':rfc_pred})
            results.index.names = ['Id']

            results.to_csv("results.csv")




