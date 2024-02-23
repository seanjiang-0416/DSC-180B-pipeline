import numpy as np
import pandas as pd
import string
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

class ClickbaitModel():
                
    def _contain_quotation_(self, text: str):
        return int("\"" in text)

    def _contain_exclamation_(self, text: str):
        return int("!" in text)

    def _contain_colon_(self, text: str):
        return int(":" in text)

    def _contain_hashtag_(self, text: str):
        return int("#" in text)

    def _contain_at_(self, text: str):
        return int("@" in text)

    def _contain_question_(self, text: str):
        if "?" in text or text.startswith(('who', 'whos', 'whose', 'what', 'whats', 'whatre', 'when', 'whens', 'whenre',
                                        'where', 'wheres', 'whered', 'wherere', 'why', 'whys', 'whyre', 'can', 'cant', 
                                        'could', 'couldnt', 'will', 'would', 'should', 'shouldnt', 'is', 'isnt',
                                        'how', 'hows', 'howre', 'howd', 'are', 'arent', 'which', 'do', 'dont', 
                                        'does', 'doesnt', 'did', 'didnt')):
            return 1
        return 0

    def _contain_please_(self, text: str):
        return int("please" in text)

    def _begin_with_digit_(self, text: str):
        if text.startswith(('1','2','3','4','5','6','7','8','9')): 
            return 1
        else: 
            return 0

    def _part_of_speech_(self, text: str):
        num_nouns = 0
        num_verbs = 0
        num_adj = 0
        num_pronouns = 0
        num_determiner = 0
        word_tokens = word_tokenize(text)

        for _, pos in pos_tag(word_tokens):
            if pos.startswith('NN'):
                num_nouns += 1
            elif pos.startswith('VB'):
                num_verbs += 1
            elif pos.startswith('JJ'):
                num_adj += 1
            elif pos.startswith('PRP'):
                num_pronouns += 1
            elif pos == 'DT':
                num_determiner += 1
        
        return num_nouns, num_verbs, num_adj, num_pronouns, num_determiner, len([w for w in word_tokens if w in self.stopwords_set]) / len(word_tokens)

    def _get_num_char_(self, text: str):
        return len(text)

    def _get_num_words_(self, text: str):
        return len(text.split())

    def _get_num_contraction_(self, text: str):
        return len([word for word in text.split() if word in self.contractions])

    def _get_average_word_length_(self, text: str):
        words = text.split()
        return sum(len(word) for word in words) / len(words)

    def _get_upper_char_ratio_(self, text: str):
        return sum(1 for c in text if c.isupper()) / self._get_num_words_(text) - 1

    def _remove_punctuation_(self, text: str):
        return text.translate(str.maketrans('', '', string.punctuation))

    def __init__(self):
        self.colNames = ['has_quotation', 'has_question', 'has_exclamation', 'has_colon',
                        'has_hashtag', 'has_at', 'upper_ratio', 'begin_with_digit', 'num_char',
                        'num_word', 'num_contra', 'average_word_length', 'num_nouns',
                        'num_verbs', 'num_adj', 'num_pronouns', 'num_determiner', 'stopwords_ratio']
        self.stopwords_set = set(stopwords.words('english'))
        self.contractions = ["im", "youre", "hes", "shes", "its", "were", "theyre",
                            "id", "youd", "hed", "shed", "itd", "wed", "theyd",
                            "ill", "youll", "hell", "shell", "itll", "well", "theyll",
                            "ive", "youve", "weve", "theyve", "dont", "doesnt", "didnt",
                            "cant", "isnt", "arent", "aint", "havent", "hasnt", "hadnt", "wasnt", "werent",
                            "wont", "wontve", "wouldnt", "shouldnt", "couldve", "shouldve", "wouldve", "mightve", "mustve",
                            "heres", "theres", "thiss", "thats", "lets", "whos", "whats", "wheres","whys", "hows",
                            "herere", "therere", "thesere", "thosere", "whore", "whatre", "wherere","whyre", "howre",
                            "herell", "therell", "thisll", "thatll", "wholl", "whatll", "wherell","whyll", "howll",
                            "hered", "thered", "thisd", "thatd", "whod", "whatd", "whered","whyd", "howd",
                            "whove", "whatve", "whereve","whyve", "howve", "mustve",
                            "gonna", "gotta", "oclock", "somebodydve", "somebodydntve", "somebodys",
                            "someoned", "someonednt", "someonedntve", "someonedve", "someonell", "someones",
                            "somethingd", "somethingdnt", "somethingdntve", "somethingdve", "somethingll", "somethings",
                            "yall", "yalldve", "yalldntve", "yallll", "yallont", "yallllve", "yallre", "yallllvent", "yaint"]

        self.data = pd.read_csv('../data/clickbait_data.csv')
        self.clf = MLPClassifier(alpha=1, max_iter=1000)
        self.scaler = StandardScaler(with_mean=False)

        X = self._embed_(self.data)
        y = self.data['clickbait']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.clf.fit(X_train, y_train)
        
        print("Training Accuracy: %f" %(self.clf.score(X_train, y_train)))
        print("Testing Accuracy: %f" %(self.clf.score(X_test, y_test)))

    def _embed_(self, df):
        df['has_quotation'] = df['headline'].apply(self._contain_quotation_)
        df['has_question'] = df['headline'].apply(self._contain_question_)
        df['has_exclamation'] = df['headline'].apply(self._contain_exclamation_)
        df['has_colon'] = df['headline'].apply(self._contain_colon_)
        df['has_hashtag'] = df['headline'].apply(self._contain_hashtag_)
        df['has_at'] = df['headline'].apply(self._contain_at_)

        df['headline'] = df['headline'].apply(self._remove_punctuation_)
        df['upper_ratio'] = df['headline'].apply(self._get_upper_char_ratio_)
        df['begin_with_digit'] = df['headline'].apply(self._begin_with_digit_)

        df['headline'] = df['headline'].str.lower()
        df['num_char'] = df['headline'].apply(self._get_num_char_)
        df['num_word'] = df['headline'].apply(self._get_num_words_)
        df['num_contra']= df['headline'].apply(self._get_num_contraction_)
        df['average_word_length'] = df['headline'].apply(self._get_average_word_length_)
        df['num_nouns'], df['num_verbs'], df['num_adj'], df['num_pronouns'], df['num_determiner'], df['stopwords_ratio'] = zip(*df['headline'].apply(self._part_of_speech_))

        return df[self.colNames]
        
    def predict(self, data):
        X = self._embed_(data)
        pred = self.clf.predict(X)
        predProb = self.clf.predict_proba(X)[:,1]
        return pred, predProb

    def predict_text(self, text: str):
        df = pd.DataFrame(index=[0], columns=['headline'])
        df['headline'] = text

        return self.predict(df)
