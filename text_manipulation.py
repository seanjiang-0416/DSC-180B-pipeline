import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import gdown
import os
import streamlit as st

class RoBERTaDetector(torch.nn.Module):
    def __init__(self, model_name="roberta-base"):
        super(RoBERTaDetector, self).__init__()
        self.bert_model = RobertaModel.from_pretrained(model_name)
        self.fc = torch.nn.Linear(self.bert_model.config.hidden_size, 2)

    def forward(self, bert_ids, bert_mask):
        _, pooler_output = self.bert_model(input_ids=bert_ids, attention_mask=bert_mask, return_dict=False)
        return self.fc(pooler_output)

def preprocess_for_prediction(text, tokenizer, max_len=256):
    tokenized_text = tokenizer.tokenize(text)[:max_len+1]
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

    pad_ind = tokenizer.pad_token_id
    input_ids = pad_sequences([token_ids], maxlen=max_len, dtype="long", truncating="post", padding="post", value=pad_ind)

    attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

    return torch.tensor(input_ids), torch.tensor(attention_masks)

def evaluate(model, input_text, tokenizer, device):
    model.eval()
    input_ids, attention_masks = preprocess_for_prediction(input_text, tokenizer)
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_masks)
        predictions = torch.argmax(outputs, dim=1)
    
    return predictions.item()

def predict(input_text):
    # Load the tokenizer and the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RoBERTaDetector().to(device)

    # Load the saved model weights
    model.load_state_dict(torch.load("models/txt_manipulation_model.pt", map_location=device))

    # Make a prediction
    prediction = evaluate(model, input_text, tokenizer, device)
    return prediction

@st.cache_data
def download_pretrained_model():
    file_id = '16HWputlJWKbu1Z_oRIDNSXKVDjvydAbD'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'models/txt_manipulation_model.pt'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    else:
        print(f"File '{output}' already exists. No download needed.")
