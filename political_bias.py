import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import nltk
from nltk.corpus import wordnet
import pickle
import gdown

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class PoliticalBiasDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def clean_text(text):
    # Check if the text is a string
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags and non-alphanumeric characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

# Tokenization function
def tokenize_function(examples):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(examples, padding='max_length', truncation=True, max_length=512)

def preprocess_article(header, content):
    # Clean the text
    cleaned_header = clean_text(header)
    cleaned_content = clean_text(content)

    # Combine header and content
    combined_text = cleaned_header + " " + cleaned_content

    # Tokenize
    encodings = tokenize_function([combined_text])

    # Create a DataLoader
    dataset = PoliticalBiasDataset(encodings, [0])  # Dummy label
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader

def predict_label(loader):
    # There might be a better way to save/load
    with open('models/poli_bias_bert.pkl', 'rb') as f:
        model = pickle.load(f)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).tolist()
        return prediction
    
def download_pretrained_model():
    file_id = '1PX2zVyPMfs0v7wxRzx7w2h-yW1h1Ti7B'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'models/poli_bias_bert.pkl'  
    gdown.download(url, output, quiet=False)
