# --- Imports ---
import streamlit as st
import re
import json
import torch
import random
from sentence_transformers import SentenceTransformer, util
from symspellpy import SymSpell
import pkg_resources
import openai
from datetime import datetime
import os

# --- Constants ---
SYSTEM_PROMPT = (
    "You are CUAB Buddy, a warm, friendly assistant for Crescent University.\n"
    "You speak clearly, casually, and helpfully like a human. Use emojis, ask clarifying questions if needed, "
    "and always try to give a complete answer. Handle student needs, academic questions, fees, and life at CUAB."
)

# --- Page Setup ---
st.set_page_config(page_title="CUAB Buddy - Crescent University Chatbot", layout="centered")
st.title("🎓 CUAB Buddy - Crescent University Chatbot")

# --- Normalization Dictionaries ---
ABBREVIATIONS = {
    "cuab": "crescent university",
    "uni": "university",
    "dept": "department",
    "admin": "admission",
    "app": "application",
    "bsc": "bachelor of science",
    "ba": "bachelor of arts",
    "phd": "doctorate"
}

SYNONYMS = {
    "courses": "programs",
    "fees": "tuition",
    "hostel": "dormitory",
    "lib": "library",
    "lecturer": "professor"
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
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

def correct_text(sym_spell, input_text):
    suggestions = sym_spell.lookup_compound(input_text, max_edit_distance=2)
    return suggestions[0].term if suggestions else input_text

# --- Load Dataset and Embed ---
def load_data_and_embed(model, path="qa_data.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = [item["question"] for item in data]
        embeddings = model.encode(questions, convert_to_tensor=True)
        return data, embeddings
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return [], None

# --- Setup Resources ---
@st.cache_resource()
def setup():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sym_spell = load_symspell()
    data, embeddings = load_data_and_embed(model)
    return model, sym_spell, data, embeddings

model, sym_spell, qa_data, qa_embeddings = setup()
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

# --- Helper Functions ---
def is_greeting(text):
    return any(greet in text.lower() for greet in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"])

def get_greeting_response():
    return random.choice([
        "Hello! 👋 How can I help you today?",
        "Hi there! Ask me anything about Crescent University 😊",
        "Welcome! What would you like to know about CUAB?"
    ])

def handle_small_talk(text):
    lower = text.lower()
    if any(kw in lower for kw in ["how are you", "what's up", "how's it going"]):
        return random.choice(["I'm doing great, thanks for asking! 😊", "All good here! What can I help you with?"])
    elif any(kw in lower for kw in ["thank you", "thanks", "appreciate"]):
        return random.choice(["You're very welcome!", "Glad I could help!", "Anytime! 😊"])
    elif "who are you" in lower or "what are you" in lower:
        return "I'm CUAB Buddy 🤖, your friendly Crescent University assistant! Ask me anything."
    return None

def resolve_follow_up(current_input):
    if st.session_state.chat_history and any(x in current_input.lower() for x in ["what about", "how about", "and the", "tell me more", "same for"]):
        last_bot = st.session_state.chat_history[-1]["bot"]
        return f"{last_bot} — {current_input}"
    return current_input

def store_in_history(user_q, bot_a):
    st.session_state.chat_history.append({"user": user_q, "bot": bot_a})

def save_to_log(user, query):
    try:
        with open("chat_log.json", "a", encoding="utf-8") as f:
            json.dump({"user": user, "query": query, "timestamp": str(datetime.now())}, f)
            f.write("\n")
    except Exception as e:
        st.error(f"Error saving log: {str(e)}")

def friendly_wrap(response):
    emojis = ["🙂", "😊", "😄", "✨", "🙌"]
    return f"{random.choice(emojis)} {response}" if not response.lower().startswith("sorry") else response

def search_answer(user_query, threshold=0.65):
    try:
        norm = normalize_text(user_query)
        corrected = correct_text(sym_spell, norm)
        user_embedding = model.encode(corrected, convert_to_tensor=True)

        if qa_embeddings is None:
            return None

        similarity_scores = util.cos_sim(user_embedding, qa_embeddings)[0]
        best_idx = torch.argmax(similarity_scores).item()
        best_score = similarity_scores[best_idx].item()

        if best_score > threshold:
            st.session_state.last_topic = qa_data[best_idx].get("topic")
            return qa_data[best_idx]["answer"]
        return None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def get_gpt_answer(prompt):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in st.session_state.chat_history[-3:]:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["bot"]})
        messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Changed to more affordable model
            temperature=0.7,
            top_p=0.9,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT error: {str(e)}")
        return "Sorry, I'm having trouble connecting to the knowledge base. Please try again later."

# --- Display Chat ---
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# --- Chat Input ---
user_input = st.chat_input("Ask me anything about Crescent University 🏫")
if user_input:
    if is_greeting(user_input):
        greeting = get_greeting_response()
        with st.chat_message("assistant"):
            st.markdown(greeting)
        store_in_history(user_input, greeting)
        save_to_log("anonymous", user_input)
    else:
        small_talk = handle_small_talk(user_input)
        if small_talk:
            with st.chat_message("assistant"):
                st.markdown(small_talk)
            store_in_history(user_input, small_talk)
            save_to_log("anonymous", user_input)
        else:
            resolved_input = resolve_follow_up(user_input)
            with st.spinner("Thinking..."):
                answer = search_answer(resolved_input)
                if answer:
                    wrapped = friendly_wrap(answer)
                    with st.chat_message("assistant"):
                        st.markdown(wrapped)
                    store_in_history(user_input, wrapped)
                    save_to_log("anonymous", user_input)
                else:
                    gpt_reply = get_gpt_answer(resolved_input)
                    with st.chat_message("assistant"):
                        st.markdown(gpt_reply)
                    store_in_history(user_input, gpt_reply)
                    save_to_log("anonymous", user_input)
