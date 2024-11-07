from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st


# Load the transformer model
@st.cache_resource
def load_transformer_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')


model = load_transformer_model()

# Define functions
def get_embeddings(texts):
    return model.encode(texts)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Streamlit UI
st.title("Text Similarity Analyzer")
text1 = st.text_area("Enter first text", "")
text2 = st.text_area("Enter second text", "")

if st.button("Calculate Similarity"):
    if text1 and text2:
        embeddings1 = get_embeddings([text1])[0]
        embeddings2 = get_embeddings([text2])[0]
        similarity = cosine_similarity(embeddings1, embeddings2)
        st.write(f"Similarity Score: {similarity:.4f}")
    else:
        st.write("Please enter text in both fields.")

#url: http://localhost:8501/