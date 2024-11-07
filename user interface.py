import streamlit as st
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf


# Load the Universal Sentence Encoder
@st.cache_resource  # Cache the model to avoid reloading on every run
def load_use_model():
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

    return hub.load(model_url)


embed = load_use_model()


# Function to get embeddings
def get_embedding(text):
    embedding = embed([text])
    return embedding.numpy()[0]


# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# Streamlit UI setup
st.title("Text Similarity Analyzer")
st.write("Enter two texts below to find out their similarity score:")

# User input for text
text1 = st.text_area("Text 1", height=150)
text2 = st.text_area("Text 2", height=150)

if st.button("Calculate Similarity"):
    if text1 and text2:
        # Get embeddings and compute similarity
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)
        similarity_score = cosine_similarity(embedding1, embedding2)

        # Display the result
        st.write(f"**Similarity Score:** {similarity_score:.2f}")
    else:
        st.write("Please enter both texts to calculate similarity.")
