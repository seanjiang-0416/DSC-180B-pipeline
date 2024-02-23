import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize
from scipy.stats import zscore
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from nltk.tokenize import word_tokenize
import torch
from nltk import sent_tokenize, pos_tag, ne_chunk


# 1 for drift, 0 for non-drift
def sentiment_score(result):
    scale = {
        'positive' : 1,
        'neutral' : 0,
        'negative' : -1
    }
    numerical_scores = [scale[sentiment['label']] * sentiment['score'] for sentiment in result[0]]
    overall_score = sum(numerical_scores)
    return overall_score

def clean_text(text):
    nltk.download('words')
    nltk.download('maxent_ne_chunker')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    cleaned_text = re.sub(r'\xa0', ' ', text)
    cleaned_text = re.sub(r'\\', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
    cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('utf-8')
    cleaned_text = cleaned_text.strip()
    cleaned_text = re.sub(r'“|”', '"', cleaned_text)
    return cleaned_text

def sentiment_shift(article):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distilled_student_sentiment_classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
        return_all_scores=True,
        device=device
    )
    cleaned_text = clean_text(article)
    data = []
    sentences = sent_tokenize(article)
    for sentence in sentences:
        # For now, trim sentence if longer than 512
        if len(sentence) > 512:
            sentence = sentence[:512]
        result = sentiment_score(distilled_student_sentiment_classifier(sentence))
        data.append(result)
    alpha = 0.05
    half = len(data)//2
    first_half = data[:half]
    second_half = data[half:]
    t_statistic, p_value = ttest_ind(first_half, second_half)
    if p_value < alpha:
        return 1
    else:
        return 0
    
# 1 for drift, 0 for non-drift
def topic_shift(article):
    cleaned_text = clean_text(article)
    data = []
    sentences = sent_tokenize(article)
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(sentences)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)
    topic_distribution = lda.transform(dtm)
    dominant_topic_per_document = topic_distribution.argmax(axis=1)
    half = len(dominant_topic_per_document) // 2
    epsilon = 1e-9
    first_half = dominant_topic_per_document[:half]
    second_half = dominant_topic_per_document[half:]
    min_value = min(min(first_half), min(second_half))
    max_value = max(max(first_half), max(second_half))
    histogram1, _ = np.histogram(first_half, bins=np.arange(min_value, max_value + 2))
    histogram2, _ = np.histogram(second_half, bins=np.arange(min_value, max_value + 2))
    contingency_table = np.array([histogram1, histogram2]) + epsilon
    _, p_value, _, _ = chi2_contingency(contingency_table)
    if p_value < 0.05:
        return 1
    else:
        return 0
    
def perform_ner(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    named_entities = ne_chunk(pos_tags)
    return [entity for entity in named_entities if isinstance(entity, nltk.Tree)]

def tree_to_string(tree):
    if isinstance(tree, nltk.Tree):
        return ' '.join([tree_to_string(child) for child in tree])
    else:
        return tree[0]
def ner_shift(article):
    sentences = sent_tokenize(article)
    sentences_length = len(sentences)
    half_index = sentences_length // 2
    first_half = ' '.join(sentences[:half_index])
    second_half = ' '.join(sentences[half_index:])
    cleaned_first_half = clean_text(first_half)
    cleaned_second_half = clean_text(second_half)
    entities_first_half = [tree_to_string(entity) for entity in perform_ner(cleaned_first_half)]
    entities_second_half = [tree_to_string(entity) for entity in perform_ner(cleaned_second_half)]
    ner_shift_count = len(set(entities_second_half) - set(entities_first_half))
    return ner_shift_count

def calculate_contextual_drift(topic_score, sent_score, ner_score):
    a = 0.4  # coefficient for topic_drift
    b = 0.4  # coefficient for sentiment_drift
    c = 0.1  # coefficient for ner_shift_count
    d = 0.1  # constant term
    score = a * topic_score + b * sent_score + c * ner_score + d
    if score >= 10:
        return 10
    return score
