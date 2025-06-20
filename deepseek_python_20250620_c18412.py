# --- Imports ---
import streamlit as st
import re
import json
import torch
import random
import time
from sentence_transformers import SentenceTransformer, util
from symspellpy import SymSpell
import pkg_resources
import openai
from datetime import datetime
import os

# --- Constants ---
SYSTEM_PROMPT = """
You are CUAB Buddy, the friendly assistant for Crescent University. Follow these rules:
1. Speak like a helpful human: use contractions ("you're", "I'll"), occasional emojis (😊), and short sentences.
2. If unsure, ask clarifying questions ("Which department are you referring to?").
3. For complex topics, break answers into bullet points.
4. Respond to greetings/small talk naturally (e.g., "Hi there! How can I help?").
5. Acknowledge emotions ("Sorry to hear that! Let me fix this...").
"""

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

# --- Setup Resources ---
@st.cache_resource()
def setup():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    data, embeddings = load_data_and_embed(model)
    return model, sym_spell, data, embeddings

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

model, sym_spell, qa_data, qa_embeddings = setup()
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

# --- Helper Functions ---
def normalize_text(text):
    text = text.lower()
    for k, v in ABBREVIATIONS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    for k, v in SYNONYMS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text

def correct_text(sym_spell, input_text):
    suggestions = sym_spell.lookup_compound(input_text, max_edit_distance=2)
    return suggestions[0].term if suggestions else input_text

def is_greeting(text):
    return any(greet in text.lower() for greet in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"])

def get_greeting_response():
    return random.choice([
        "Hello! 👋 How can I help you today?",
        "Hi there! Ask me anything about Crescent University 😊",
        "Welcome! What would you like to know about CUAB?"
    ])

def detect_emotion(text):
    text = text.lower()
    if any(word in text for word in ["angry", "frustrated", "annoyed"]):
        return "frustrated"
    elif any(word in text for word in ["thank", "appreciate", "helpful"]):
        return "grateful"
    return None

def format_response(response, emotion):
    if emotion == "frustrated":
        return f"I’m really sorry about that! 😔 Let’s resolve this: {response}"
    elif emotion == "grateful":
        return f"Happy to help! 💙 {response}"
    return response

def handle_small_talk(text):
    lower = text.lower()
    if any(kw in lower for kw in ["how are you", "what's up", "how's it going"]):
        return random.choice([
            "Doing great! Ready to help you with CUAB stuff. 😊",
            "Just here, waiting to assist you! What’s up?"
        ])
    elif any(kw in lower for kw in ["thank you", "thanks", "appreciate"]):
        return random.choice([
            "You’re welcome! 💙",
            "Anytime! 😄",
            "Glad I could help!"
        ])
    elif "your name" in lower:
        return "I’m CUAB Buddy! (But you can call me Buddy 😉)"
    return None

def suggest_follow_up(topic):
    suggestions = {
        "admission": "Would you like details about required documents or deadlines?",
        "fees": "Need help with payment methods or installment plans?",
        "courses": "Want to check related courses or lecturers?",
        "hostel": "Should I explain the hostel application process?",
    }
    return suggestions.get(topic, "Is there anything else I can help with? 😊")

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

def stream_response(response):
    message_placeholder = st.empty()
    full_response = ""
    for chunk in response.split():
        full_response += chunk + " "
        time.sleep(0.05)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)

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

def get_gpt_answer(user_query):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[{"role": "user" if i % 2 == 0 else "assistant", "content": msg["user"] if i % 2 == 0 else msg["bot"]} 
              for i, msg in enumerate(st.session_state.chat_history[-4:])],
            {"role": "user", "content": user_query}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            presence_penalty=0.5
        )
        
        answer = response.choices[0].message.content
        
        if st.session_state.last_topic:
            follow_up = suggest_follow_up(st.session_state.last_topic)
            answer += f"\n\n{follow_up}"
            
        return answer
    except Exception as e:
        return random.choice([
            "I’m having connection issues. Could you try again?",
            "Oops! My wires got tangled. Ask me once more?"
        ])

def get_answer(user_query):
    # 1. Check greetings
    if is_greeting(user_query):
        return get_greeting_response()
    
    # 2. Check small talk
    small_talk_response = handle_small_talk(user_query)
    if small_talk_response:
        return small_talk_response
    
    # 3. Resolve follow-up context
    resolved_input = resolve_follow_up(user_query)
    
    # 4. Search QA database
    qa_answer = search_answer(resolved_input)
    if qa_answer:
        return qa_answer
    
    # 5. Fallback to GPT
    gpt_answer = get_gpt_answer(resolved_input)
    if "sorry" in gpt_answer.lower() or "don't know" in gpt_answer.lower():
        return random.choice([
            "I’m not 100% sure. Could you rephrase that?",
            "Hmm, I might need more details. Can you elaborate?"
        ])
    return gpt_answer

# --- UI Setup ---
st.set_page_config(page_title="CUAB Buddy - Crescent University Chatbot", layout="centered")
st.title("🎓 CUAB Buddy - Crescent University Chatbot")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# Chat input
user_input = st.chat_input("Ask me anything about CUAB...")
if user_input:
    # Detect emotion
    emotion = detect_emotion(user_input)
    
    # Get response
    response = get_answer(user_input)
    response = format_response(response, emotion)
    
    # Store history
    store_in_history(user_input, response)
    save_to_log("anonymous", user_input)
    
    # Display
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        stream_response(response)