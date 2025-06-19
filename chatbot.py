import streamlit as st
import torch
from utils import normalize_text, correct_text, load_symspell, load_data_and_embed
from sentence_transformers import SentenceTransformer, util
import openai
import json

# Load model and data
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")

@st.cache_resource()
def setup():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sym_spell = load_symspell()
    data, embeddings = load_data_and_embed(model)
    return model, sym_spell, data, embeddings

model, sym_spell, qa_data, qa_embeddings = setup()

openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "sk-..."

def search_answer(user_query, threshold=0.65):
    norm = normalize_text(user_query)
    corrected = correct_text(sym_spell, norm)
    user_embedding = model.encode(corrected, convert_to_tensor=True)
    similarity_scores = util.cos_sim(user_embedding, qa_embeddings)[0]
    best_idx = torch.argmax(similarity_scores).item()
    best_score = similarity_scores[best_idx].item()

    if best_score > threshold:
        return qa_data[best_idx]["answer"]
    else:
        return None

def get_gpt_answer(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, GPT is currently unavailable."

# Chat Interface
user_input = st.text_input("Ask me anything about Crescent University 🏫", key="input")
if user_input:
    with st.spinner("Thinking..."):
        answer = search_answer(user_input)
        if answer:
            st.success(answer)
        else:
            gpt_reply = get_gpt_answer(user_input)
            st.info(gpt_reply)
