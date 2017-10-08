import pandas as pd

from preprocessor import Preprocessor
from tf import TF
from tfidf import TFIDF
from training import Training

print("> Loading training set")

X = pd.read_csv("train_set_x.csv")
y = pd.read_csv("train_set_y.csv")

# X = X.truncate(after=10000)
# y = y.truncate(after=10000)

X.drop('Id', axis=1, inplace=True)
y.drop('Id', axis=1, inplace=True)

preprocessor = Preprocessor()
preprocessor.process(X, inplace=True)

print("> Creating feature extraction pipeline")

feature_extraction_pipeline = []
feature_extraction_pipeline.append(TF(X))
feature_extraction_pipeline.append(TFIDF(X, category_df=y))

print("> Extracting features from training set")

# Use the pipeline to extract features
for feature_extractor in feature_extraction_pipeline:
    extracted_features = feature_extractor.extractFeatures()
    X = pd.concat([X, feature_extractor.addPrefix(extracted_features)], axis=1)

# No need for text column anymore, since the features were extracted
X.drop('Text', axis=1, inplace=True)

Training(X,y,feature_extraction_pipeline).train(validation=False)

print("> Done")

