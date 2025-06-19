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

# --- Constants ---
SYSTEM_PROMPT = (
    "You are a warm, friendly assistant for Crescent University. "
    "You speak clearly, casually, and helpfully like a human. You use emojis, ask clarifying questions if needed, "
    "and always try to give a complete answer."
)

# --- Page Setup ---
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")

# --- Normalization Dictionaries ---
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could",
    "shud": "should", "wud": "would", "abt": "about", "bcz": "because",
    "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester",
    "admsn": "admission", "clg": "college", "sch": "school", "uni": "university",
    "cresnt": "crescent", "l": "level", "d": "the", "msg": "message",
    "idk": "i don't know", "imo": "in my opinion", "asap": "as soon as possible",
    "dept": "department", "reg": "registration", "fee": "fees", "pg": "postgraduate",
    "app": "application", "req": "requirement", "nd": "national diploma",
    "a-level": "advanced level", "alevel": "advanced level", "2nd": "second",
    "1st": "first", "nxt": "next", "prev": "previous", "exp": "experience",
    "csc": "department of computer science", "mass comm": "department of mass communication",
    "law": "department of law", "acc": "department of accounting"
}

SYNONYMS = {
    "lecturers": "academic staff", "professors": "academic staff",
    "teachers": "academic staff", "instructors": "academic staff",
    "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department",
    "school": "university", "college": "faculty", "course": "subject",
    "class": "course", "subject": "course", "unit": "credit",
    "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "admin worker": "non-academic staff", "support staff": "non-academic staff",
    "clerk": "non-academic staff", "receptionist": "non-academic staff",
    "secretary": "non-academic staff", "tech staff": "technical staff",
    "hostel": "accommodation", "lodging": "accommodation", "room": "accommodation",
    "school fees": "tuition", "acceptance fee": "admission fee", "fees": "tuition",
    "enrol": "apply", "join": "apply", "sign up": "apply", "admit": "apply",
    "requirement": "criteria", "conditions": "criteria", "needed": "required",
    "needed for": "required for", "who handles": "who manages"
}

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

# --- Memory State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(greet in text.lower() for greet in greetings)

def get_greeting_response():
    return random.choice([
        "Hello! 👋 How can I help you today?",
        "Hi there! Ask me anything about Crescent University 😊",
        "Welcome! What would you like to know about CUAB?"
    ])

def handle_small_talk(text):
    lower = text.lower()
    if any(kw in lower for kw in ["how are you", "what’s up", "how’s it going"]):
        return random.choice(["I'm doing great, thanks for asking! 😊", "All good here! What can I help you with?"])
    elif any(kw in lower for kw in ["thank you", "thanks", "appreciate"]):
        return random.choice(["You're very welcome!", "Glad I could help!", "Anytime! 😊"])
    elif "who are you" in lower or "what are you" in lower:
        return "I'm your friendly Crescent University assistant bot 🤖 here to guide you through anything CUAB!"
    return None

def resolve_follow_up(current_input):
    if any(x in current_input.lower() for x in ["what about", "how about", "and the", "tell me more", "same for"]):
        if st.session_state.chat_history:
            last_bot = st.session_state.chat_history[-1]["bot"]
            return f"{last_bot} — {current_input}"
    return current_input

def store_in_history(user_q, bot_a):
    st.session_state.chat_history.append({"user": user_q, "bot": bot_a})

def save_to_log(user, query):
    with open("chat_log.json", "a", encoding="utf-8") as f:
        json.dump({"user": user, "query": query}, f)
        f.write("\n")

def friendly_wrap(response):
    emojis = ["🙂", "😊", "😄", "✨", "🙌"]
    return f"{random.choice(emojis)} {response}" if not response.lower().startswith("sorry") else response

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

# --- Display Conversation History ---
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# --- Streamlit UI ---
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

# --- Debugging ---
if st.checkbox("🔍 Show Debug Info"):
    st.write("Normalized Input:", normalize_text(user_input))
    st.write("Corrected Input:", correct_text(sym_spell, user_input))

