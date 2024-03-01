import warnings
warnings.filterwarnings('ignore')

import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class SourceReliableModel():

    def _clean_(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'\n', '', text)
        text = re.sub(r'[^\w\s]', '', text)

        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(self.stemmer.stem(word)) for word in words if word not in self.stopwords_set]

        return ' '.join(words)

    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_set = set(stopwords.words('english'))

        tfidfV = TfidfVectorizer(stop_words='english')
        self.pipeline = Pipeline([
                ('tfidf', tfidfV),
                ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
                ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
    
    def score(self, X, y):
        return self.pipeline.score(X, y)

    def predict(self, X:pd.DataFrame):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X:pd.DataFrame):
        return self.pipeline.predict_proba(X)[:,1]
    
    def predict_text(self, text):
        clean_text = self._clean_(text)
        pred = self.pipeline.predict([clean_text])
        predProb = self.pipeline.predict_proba([clean_text])[:, 1]
        return pred[0], predProb[0]