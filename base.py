import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from preprocessor import Preprocessor
from tf import TF
from tfidf import TFIDF

print("> Loading training set")

X = pd.read_csv("train_set_x.csv")
y = pd.read_csv("train_set_y.csv")

# X = X.truncate(after=10000)
# y = y.truncate(after=10000)

X.drop('Id', axis=1, inplace=True)
y.drop('Id', axis=1, inplace=True)

X_train = X
y_train = y

preprocessor = Preprocessor()
preprocessor.process(X_train, inplace=True)

print("> Creating feature extraction pipeline")

feature_extraction_pipeline = []
feature_extraction_pipeline.append(TF(X_train))
feature_extraction_pipeline.append(TFIDF(X_train, category_df=y_train))

print("> Extracting features from training set")

# Use the pipeline to extract features
for feature_extractor in feature_extraction_pipeline:
    extracted_features = feature_extractor.extractFeatures()
    X_train = pd.concat([X_train, feature_extractor.addPrefix(extracted_features)], axis=1)

print("> Training the model")

# No need for text column anymore, since the features were extracted
X_train = X_train.drop('Text', axis=1)

# Strat classification
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train.values.ravel())

print("> Setting up the test set")

# Apply feature extraction to test set
X_test = pd.read_csv('test_set_x.csv')
X_test.drop('Id', axis=1, inplace=True)
preprocessor.process(X_test, inplace=True)

print("> Getting the features from test set")

for feature_extractor in feature_extraction_pipeline:
    applied_features = feature_extractor.applyToTest(X_test)
    X_test = pd.concat([X_test, feature_extractor.addPrefix(applied_features)], axis=1)

# No need for the text column anymore
X_test = X_test.drop('Text', axis=1)

print("> Predicting")

rfc_pred = rfc.predict(X_test)

print("> Exporting")

results = pd.DataFrame({'Category':rfc_pred})
results.index.names = ['Id']

results.to_csv("results.csv")

print("> Done")

