# This version includes improvements for all 8 steps

# --- Imports ---
import streamlit as st
import re
import json
import torch
import random
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import pkg_resources
import openai
from datetime import datetime

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
ABBREVIATIONS = {...}  # Same as before
SYNONYMS = {...}        # Same as before

# --- Normalization Functions ---
def normalize_text(text):
    text = text.lower()
    for k, v in ABBREVIATIONS.items():
        text = re.sub(rf"\\b{k}\\b", v, text)
    for k, v in SYNONYMS.items():
        text = re.sub(rf"\\b{k}\\b", v, text)
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
openai.api_key = st.secrets.get("OPENAI_API_KEY", "sk-...")

# --- Session State ---
for key in ["chat_history", "last_topic"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# --- UI Filters ---
faculty = st.selectbox("Faculty", ["", "Law", "Health Sciences", "Natural and Applied Sciences", "Arts and Social Sciences", "Environmental Sciences"])
department = st.selectbox("Department", ["", "Computer Science", "Mass Communication", "Nursing", "Accounting", "Architecture"])
level = st.selectbox("Level", ["", "100", "200", "300", "400", "500"])
semester = st.selectbox("Semester", ["", "First", "Second"])

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
    if any(x in current_input.lower() for x in ["what about", "how about", "and the", "tell me more", "same for"]):
        last_bot = st.session_state.chat_history[-1]["bot"] if st.session_state.chat_history else ""
        return f"{last_bot} — {current_input}"
    return current_input

def store_in_history(user_q, bot_a):
    st.session_state.chat_history.append({"user": user_q, "bot": bot_a})

def save_to_log(user, query):
    with open("chat_log.json", "a", encoding="utf-8") as f:
        json.dump({"user": user, "query": query, "timestamp": str(datetime.now())}, f)
        f.write("\n")

def friendly_wrap(response):
    emojis = ["🙂", "😊", "😄", "✨", "🙌"]
    return f"{random.choice(emojis)} {response}" if not response.lower().startswith("sorry") else response

def search_answer(user_query, threshold=0.65):
    norm = normalize_text(user_query)
    corrected = correct_text(sym_spell, norm)
    user_embedding = model.encode(corrected, convert_to_tensor=True)

    filtered_data = []
    filtered_embeddings = []
    for i, item in enumerate(qa_data):
        if (not faculty or item.get("faculty") == faculty) and \
           (not department or item.get("department") == department) and \
           (not level or item.get("level") == level) and \
           (not semester or item.get("semester") == semester):
            filtered_data.append(item)
            filtered_embeddings.append(qa_embeddings[i])

    if not filtered_data:
        return None

    similarity_scores = util.cos_sim(user_embedding, torch.stack(filtered_embeddings))[0]
    best_idx = torch.argmax(similarity_scores).item()
    best_score = similarity_scores[best_idx].item()

    if best_score > threshold:
        st.session_state.last_topic = filtered_data[best_idx].get("topic")
        return filtered_data[best_idx]["answer"]
    return None

def get_gpt_answer(prompt):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in st.session_state.chat_history[-3:]:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["bot"]})
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.7,
            top_p=0.9,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Sorry, GPT is currently unavailable."

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
        st.success(greeting)
        store_in_history(user_input, greeting)
        save_to_log("anonymous", user_input)
    else:
        small_talk = handle_small_talk(user_input)
        if small_talk:
            st.info(small_talk)
            store_in_history(user_input, small_talk)
            save_to_log("anonymous", user_input)
        else:
            resolved_input = resolve_follow_up(user_input)
            with st.spinner("Thinking..."):
                answer = search_answer(resolved_input)
                if answer:
                    wrapped = friendly_wrap(answer)
                    st.success(wrapped)
                    store_in_history(user_input, wrapped)
                    save_to_log("anonymous", user_input)
                else:
                    gpt_reply = get_gpt_answer(resolved_input)
                    st.info(gpt_reply)
                    store_in_history(user_input, gpt_reply)
                    save_to_log("anonymous", user_input)
