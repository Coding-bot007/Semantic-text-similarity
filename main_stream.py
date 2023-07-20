import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle

# Load the pre-trained BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Load your pretrained semantic text similarity model
with open('semantics_model.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# Function to compute the similarity score
def compute_similarity(text1, text2):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokens1 = tokenizer(text1, return_tensors="pt")
    tokens2 = tokenizer(text2, return_tensors="pt")

    model = BertModel.from_pretrained("bert-base-uncased")
    with torch.no_grad():
        output1 = model(**tokens1)
        output2 = model(**tokens2)

    sentence_embedding1 = torch.mean(output1.last_hidden_state, dim=1)  # Average pooling
    sentence_embedding2 = torch.mean(output2.last_hidden_state, dim=1)

    cosine_sim = cosine_similarity(sentence_embedding1.numpy(), sentence_embedding2.numpy())

    # Return the similarity score instead of printing it
    return cosine_sim[0][0]


# Streamlit app code
st.title("Semantic Text Similarity Prediction")

st.write("Enter two sentences to find their similarity:")

# Input text areas for text1 and text2
text1 = st.text_area("Enter Text 1:")
text2 = st.text_area("Enter Text 2:")

# Prediction button
if st.button("Predict Similarity"):
    if text1 and text2:
        similarity_score = compute_similarity(text1, text2)
        st.json({"text1": text1, "text2": text2, "similarity score": similarity_score})
    else:
        st.warning("Please enter both Text 1 and Text 2 for prediction.")
