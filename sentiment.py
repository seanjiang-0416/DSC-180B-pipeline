import string
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

class SentimentModel():
    def _cleaning_(self, raw_text: str):
        text = re.sub("[^a-zA-Z]", " ", raw_text)
        text =  raw_text.lower()
        news_words = word_tokenize(text)
        words = [w for w in  news_words  if not w in self.stopwords_set]
        lem = [ WordNetLemmatizer().lemmatize(w) for w in words ]
        stems = [self.stemmer.stem(w) for w in lem ]
        return " ".join(stems)

    def _compute_polarity_(self, text: str):
        scores = self.senti.polarity_scores(text)
        if(scores['neg'] > scores['pos'] and scores['neg'] > scores['neu']):
            return -1
        elif(scores['pos'] > scores['neg'] and scores['pos'] > scores['neu']):
            return 1
        else:
            return 0

    def __init__(self):
        self.stopwords_set = set(stopwords.words('english'))
        self.senti = SentimentIntensityAnalyzer()
        self.stemmer = SnowballStemmer('english')

        tfidfV = TfidfVectorizer(stop_words='english')
        self.pipeline = Pipeline([
                ('tfidf', tfidfV),
                ('log_clf',LogisticRegression(solver='liblinear', C=0.1))
                ])

    def fit(self, data_train):
        data_train['clean_headline'] = data_train['headline'].apply(self._cleaning_)
        X_train = data_train['clean_headline'] + '. ' + data_train['context'].apply(self._cleaning_)
        y_train = data_train['clean_headline'].apply(self._compute_polarity_)
        
        self.pipeline.fit(X_train,y_train)
        print("Training Accuracy: %f" %(self.pipeline.score(X_train, y_train)))

    def predict(self, data:pd.DataFrame):
        X = data['headline'].apply(self._cleaning_) + '. ' + data['context'].apply(self._cleaning_)
        pred = self.pipeline.predict(X)
        predProb = self.pipeline.predict_proba(X)[:,1]
        return pred, predProb

    def predict_article(self, headline: str, context: str):
        cleaned_headline = self._cleaning_(headline)
        cleaned_context = self._cleaning_(context)
        combined_text = cleaned_headline + '. ' + cleaned_context
        pred = self.pipeline.predict([combined_text])
        predProb = self.pipeline.predict_proba([combined_text])[:, 1]
        return pred[0], predProb[0]