import html
import re
import requests
import torch
from transformers import BertTokenizer, BertModel, pipeline
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def text_embedding(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    def get_bert_embeddings(data):
        tokens = tokenizer(data.tolist(), padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            embeddings = bert_model(**tokens).last_hidden_state.mean(dim=1)
        return embeddings

    batch_size = 128
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    embeddings_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = data.iloc[start_idx:end_idx]
        batch_embeddings = get_bert_embeddings(batch_data)
        embeddings_list.append(batch_embeddings)

    embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()
    return embeddings

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    tokens = nltk.word_tokenize(text)    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def search_wikipedia(query, num_results=15):
    endpoint_url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': query,
        'srlimit': num_results
    }
    response = requests.get(endpoint_url, params=params)
    data = response.json()

    output = []
    search_results = data['query']['search']
    for result in search_results:
        title = result['title']
        snippet = html.unescape(result['snippet'])
        clean_snippet = re.sub('<.*?>', '', snippet)
        output.append(title + ": " + clean_snippet)

    return " ".join(output)

