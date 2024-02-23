import string
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class SourceReliableModel():

    def _cleaning_(self, raw_text: str):
        text = re.sub("[^a-zA-Z]", " ", raw_text)
        text =  text.lower()
        news_words = word_tokenize(text)
        words = [w for w in  news_words  if not w in self.stopwords_set]
        lem = [ WordNetLemmatizer().lemmatize(w) for w in words ]
        stems = [self.stemmer.stem(w) for w in lem ]
        return " ".join(stems)

    def __init__(self):
        self.stopwords_set = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')

        tfidfV = TfidfVectorizer(stop_words='english')
        self.pipeline = Pipeline([
                ('tfidf', tfidfV),
                ('log_clf', LogisticRegression(solver='liblinear', C=0.3))
                ])

    def fit(self, data_train):
        df = data_train[['headline', 'context', 'src_true','src_mostly_true', 'src_half_true', 'src_mostly_false', 'src_false', 'src_pants_on_fire']]
        X_train = df['headline'].apply(self._cleaning_) + '. ' + df['context'].apply(self._cleaning_)
        y_train = np.argmax(df.iloc[:, 2:].values, axis = 1)
        
        self.pipeline.fit(X_train,y_train)
        print("Training Accuracy: %f" %(self.pipeline.score(X_train, y_train)))

    def predict(self, data:pd.DataFrame):
        X = data['headline'].apply(self._cleaning_) + '. ' + data['context'].apply(self._cleaning_)
        pred = self.pipeline.predict(X)
        predProb = self.pipeline.predict_proba(X)[:,1]
        return pred, predProb