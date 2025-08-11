from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
import pickle as pkl
import torch
import numpy as np

def find_class(tokenizer, model, logit_model, text):
    label_map = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science/Technology"
    }

    texts = [text]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs = {key: value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        batch_embeddings = last_hidden_state.mean(dim=1).numpy()
    outputs = logit_model.predict(batch_embeddings)
    return label_map[outputs[0]]

def save_models():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    logit_model = None

    with open("logistic_regression_model.pkl", "rb") as f:
        logit_model = pkl.load(f)
    
    with open("all_models.pkl", "wb") as f:
        pkl.dump([tokenizer, model, logit_model], f)

def load_models():
    tokenizer, model, logit_model = None, None, None
    with open("all_models.pkl", "rb") as f:
        tokenizer, model, logit_model = pkl.load(f)
    return tokenizer, model, logit_model


if __name__ == "__main__":
    # save_models()
    text = "Best Asian Tourism Destinations The new APMF survey of the best Asian tourism destinations has just kicked off, but it's crowded at the top, with Chiang Mai in Thailand just leading from perennial favourites Hong Kong, Bangkok and Phuket in Thailand, and Bali in Indonesia. Be one of the first to vote and let us know your reasons."
    tokenizer, model, logit_model = load_models()
    print(find_class(tokenizer, model, logit_model, text))