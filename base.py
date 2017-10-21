import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from knearestneighbor import KNearestNeighbor
from naivebayesclassifier import NaiveBayesClassifier

from preprocessor import Preprocessor
from tf import TF
from tfidf import TFIDF
from training import Training

print("> Loading training set")

X = pd.read_csv("train_set_x.csv")
y = pd.read_csv("train_set_y.csv")

#X = X.truncate(after=10000)
#y = y.truncate(after=10000)

X.drop('Id', axis=1, inplace=True)
y.drop('Id', axis=1, inplace=True)

preprocessor = Preprocessor()
preprocessor.process(X, inplace=True)

X.to_csv('cleanedup.csv', encoding="utf-8")

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

# Training(X,y,feature_extraction_pipeline, RandomForestClassifier(n_estimators=200, verbose=3)).train(validation=True, featureSelect = False)
#Training(X,y,feature_extraction_pipeline, ExtraTreesClassifier(n_estimators=300, verbose=3, n_jobs=-1)).train(validation=True,num_features=200,random_state=2)
#Training(X,y,feature_extraction_pipeline, KNearestNeighbor(K=7)).train(validation=True, num_features=20)
Training(X,y,feature_extraction_pipeline, NaiveBayesClassifier()).train(validation=True, num_features=250, random_state=8)


print("> Done")
