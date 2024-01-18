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

# Load the dataset
file_path = 'allside_articles.csv'
data = pd.read_csv(file_path)

# Adjusting the cleaning function to handle non-string values
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

# Reapply the cleaning function
data['Header_clean'] = data['Header'].apply(clean_text)
data['Content_clean'] = data['Content'].apply(clean_text)

# Recheck for missing values
missing_values_updated = data[['Header_clean', 'Content_clean', 'Label']].isnull().sum()

import random
import nltk
from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in nltk.corpus.stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: # only replace up to n words
            break

    sentence = ' '.join(new_words)
    return sentence

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char.isalpha()])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# Simple preprocessing
data['text'] = data['Header_clean'] + " " + data['Content_clean']  # Combining header and content
data = data[['text', 'Label']]  

# Step 1: Generate Augmented Text
augmented_texts_5 = [synonym_replacement(sentence, 5) for sentence in data['text']]
augmented_texts_10 = [synonym_replacement(sentence, 10) for sentence in data['text']]
augmented_texts_30 = [synonym_replacement(sentence, 30) for sentence in data['text']]

# Step 2: Create a new DataFrame with augmented text
augmented_data_5 = pd.DataFrame({'text': augmented_texts_5, 'Label': data['Label']})
augmented_data_10 = pd.DataFrame({'text': augmented_texts_10, 'Label': data['Label']})
augmented_data_30 = pd.DataFrame({'text': augmented_texts_30, 'Label': data['Label']})

# Combine the original and augmented datasets
combined_data = pd.concat([data, augmented_data_5, augmented_data_10, augmented_data_30], axis=0)

# Step 3: Shuffle the combined dataset
combined_data = combined_data.sample(frac=1).reset_index(drop=True)


# Try with larger dataset
data = combined_data

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode the labels
data['Label'] = label_encoder.fit_transform(data['Label'])

# Split the data (with encoded labels)
train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['Label'], test_size=0.2)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples, padding='max_length', truncation=True, max_length=512)

# Apply the tokenizer to the dataset
train_encodings = tokenize_function(train_texts.tolist())
val_encodings = tokenize_function(val_texts.tolist())

# Dataset class
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

# Convert to dataset format
train_dataset = PoliticalBiasDataset(train_encodings, train_labels.tolist())
val_dataset = PoliticalBiasDataset(val_encodings, val_labels.tolist())

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['Label'].unique()))

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-6)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

# Move model to GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
model.train()
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
# Could be a better way to save/load        
with open('poli_bias_bert.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluation
model.eval()
predictions = []
references = []
for batch in val_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=-1).tolist())
    references.extend(labels.tolist())

# Classification report
print(classification_report(references, predictions))