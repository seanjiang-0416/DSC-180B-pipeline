import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pickle


binary_fp = os.path.join("cleaned_binary.csv") #data is from Google's Big Bench dataset, mapped to fallacy vs non-fallacy

binary = pd.read_csv(binary_fp)
binary.replace({'Invalid': 0, 'Valid': 1}, inplace = True)

X, y = binary['input'], binary['label']


X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.2)

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)

fallacy_model = AdaBoostClassifier() #obtains 71% accuracy on test set
fallacy_model.fit(X_train, y_train)

with open('adaboost_model.pkl', 'wb') as file:
    pickle.dump(fallacy_model, file)

