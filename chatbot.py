# --- Imports ---
import streamlit as st
import re
import json
import torch
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import pkg_resources
import openai

# --- Page Setup ---
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")

# --- Normalization Dictionaries ---
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could",
    "abt": "about", "b4": "before", "info": "information"
}

SYNONYMS = {
    "it people": "technical staff",
    "office staff": "non-academic staff",
    "lecturers": "academic staff",
    "school fees": "tuition"
}

# --- Normalization Functions ---
def normalize_text(text):
    text = text.lower()
    for k, v in ABBREVIATIONS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    for k, v in SYNONYMS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text

def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

def correct_text(sym_spell, input_text):
    suggestions = sym_spell.lookup_compound(input_text, max_edit_distance=2)
    return suggestions[0].term if suggestions else input_text

# --- Load Dataset and Embed ---
def load_data_and_embed(model, path="qa_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return data, embeddings

# --- Setup Resources ---
@st.cache_resource()
def setup():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sym_spell = load_symspell()
    data, embeddings = load_data_and_embed(model)
    return model, sym_spell, data, embeddings

model, sym_spell, qa_data, qa_embeddings = setup()

# --- OpenAI Key (Replace or use Streamlit secrets) ---
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "sk-..."

# --- Search Logic ---
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
    except Exception:
        return "Sorry, GPT is currently unavailable."

# --- Streamlit UI ---
user_input = st.text_input("Ask me anything about Crescent University 🏫", key="input")

if user_input:
    with st.spinner("Thinking..."):
        answer = search_answer(user_input)
        if answer:
            st.success(answer)
        else:
            gpt_reply = get_gpt_answer(user_input)
            st.info(gpt_reply)
