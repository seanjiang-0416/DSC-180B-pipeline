import string
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

class SpamModel():

    def _cleaning_(self, raw_text: str):
        text = re.sub("[^a-zA-Z]", " ", raw_text)
        text =  text.lower()
        news_words = word_tokenize(text)
        words = [w for w in  news_words  if not w in self.stopwords_set]
        lem = [ WordNetLemmatizer().lemmatize(w) for w in words ]
        stems = [self.stemmer.stem(w) for w in lem ]
        return " ".join(stems)
        
    def _encode_target_(self, input: str):
        if input < 4:
                return 'true'
        else:
                return 'false'
        
    def __init__(self):
        self.stopwords_set = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')

        spam_file = open("../data/spamwords.txt", "r") 
        data = spam_file.read() 
        self.spamword_lst = data.split("\n") 
        spam_file.close() 

        self.countV = CountVectorizer(vocabulary=set(self.spamword_lst))
        self.pipeline = Pipeline([
                ('spamCnt', self.countV),
                ('knn',KNeighborsClassifier(3))
                ])

    def fit(self, data_train):
        X_train = data_train['headline'].apply(self._cleaning_)
        y_train = data_train['target'].apply(self._encode_target_)
        
        self.pipeline.fit(X_train,y_train)
        print("Training Accuracy: %f" %(self.pipeline.score(X_train, y_train)))


    def predict(self, data:pd.DataFrame):
        X = data['headline'].apply(self._cleaning_)
        pred = self.pipeline.predict(X)
        predProb = self.pipeline.predict_proba(X)[:,1]
        return pred, predProb
    
    def predict_article(self, headline: str):
        cleaned_headline = self._cleaning_(headline)
        pred = self.pipeline.predict([cleaned_headline])
        pred_prob = self.pipeline.predict_proba([cleaned_headline])[:, 1]
        return pred[0], pred_prob[0]
