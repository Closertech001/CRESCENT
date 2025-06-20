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
import inflect

# --- Initialize Inflect Engine ---
p = inflect.engine()

# --- Constants ---
SYSTEM_PROMPT = """
You are CUAB Buddy, the friendly assistant for Crescent University. Follow these rules:
1. Speak naturally like a helpful human - use contractions ("you're", "I'll"), occasional emojis (😊), and varied sentence lengths.
2. Show personality - be warm, occasionally humorous, and empathetic.
3. If unsure, ask clarifying questions in a natural way ("Which department did you mean?").
4. For complex topics, break answers into digestible parts with bullet points when helpful.
5. Mirror the user's tone - if they're formal, be slightly formal; if casual, be friendly.
6. Remember context from the conversation but don't over-rely on it.
7. Admit when you don't know something but offer to help find the answer.
"""

# --- Enhanced Normalization Dictionaries ---
VERB_CONJUGATIONS = {
    "does": "do", "has": "have", "was": "be", "were": "be",
    "did": "do", "are": "be", "is": "be", "studies": "study",
    "teaches": "teach", "offers": "offer", "requires": "require"
}

PLURAL_MAP = {
    "courses": "course", "subjects": "subject", "lecturers": "lecturer",
    "departments": "department", "faculties": "faculty", "fees": "fee",
    "requirements": "requirement", "books": "book", "students": "student"
}

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

DEPARTMENT_ALIASES = {
    "anatomy": ["ana", "dept of anatomy", "anatomy dept"],
    "accounting": ["acc", "dept of accounting", "accounting dept"],
    "architecture": ["arch", "dept of architecture", "architecture dept"],
    "computer science": ["cs", "csc", "dept of computer science", "computer science dept"],
    "law": ["bacolaw", "dept of law", "law dept"],
    "nursing": ["nur", "dept of nursing", "nursing dept"],
    "physiology": ["phy", "dept of physiology", "physiology dept"],
    "mass communication": ["mass comm", "dept of mass communication", "mass comm dept"],
    "economics": ["eco", "dept of economics", "economics dept"],
    "business administration": ["bus admin", "dept of business admin", "business admin dept"],
    "biochemistry": ["bch", "dept of biochemistry", "biochemistry dept"],
    "microbiology": ["mcb", "dept of microbiology", "microbiology dept"],
    "political science": ["pol sci", "dept of political science", "political science dept"]
}

# --- Setup Resources ---
@st.cache_resource()
def setup_resources():
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

# Initialize resources
model, sym_spell, qa_data, qa_embeddings = setup_resources()
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {
        "current_department": None,
        "current_topic": None,
        "user_mood": "neutral"
    }

# --- Enhanced Helper Functions ---
def normalize_text(text):
    """Enhanced text normalization with course code preservation"""
    text = text.lower().strip()
    
    # Preserve course codes (GST 111 → gst111)
    text = re.sub(r'([a-z]{2,4})[\s-]?(\d{3})', r'\1\2', text)
    
    # Convert plurals to singular
    words = []
    for word in text.split():
        if word in VERB_CONJUGATIONS:
            words.append(VERB_CONJUGATIONS[word])
        elif word in PLURAL_MAP:
            words.append(PLURAL_MAP[word])
        else:
            singular = p.singular_noun(word)
            words.append(singular if singular else word)
    
    text = " ".join(words)
    
    # Apply abbreviations and synonyms
    for k, v in {**ABBREVIATIONS, **SYNONYMS}.items():
        text = re.sub(rf"\b{k}\b", v, text)
    
    # Handle department aliases
    for dept, aliases in DEPARTMENT_ALIASES.items():
        for alias in aliases:
            text = re.sub(rf"\b{alias}\b", dept, text)
    
    return text

def detect_emotion(text):
    """Detect user's emotional state from text"""
    text = text.lower()
    positive_words = ["thank", "appreciate", "helpful", "great", "awesome", "perfect"]
    negative_words = ["angry", "frustrated", "annoyed", "upset", "disappointed", "sucks"]
    
    if any(word in text for word in positive_words):
        return "positive"
    elif any(word in text for word in negative_words):
        return "negative"
    return "neutral"

def get_greeting_response():
    """Return a natural, varied greeting response"""
    greetings = [
        "Hi there! 😊 How can I help you today?",
        "Hello! Welcome to CUAB Buddy. What can I do for you?",
        "Hey! 👋 Great to see you. What's on your mind?",
        "Good day! How can I assist you with Crescent University matters?"
    ]
    return random.choice(greetings)

def handle_small_talk(text):
    """Handle casual conversation naturally"""
    lower = text.lower()
    
    if any(kw in lower for kw in ["how are you", "what's up", "how's it going"]):
        return random.choice([
            "I'm doing well, thanks for asking! Ready to help you with anything CUAB-related. 😊",
            "Just here and excited to help you! What can I do for you today?"
        ])
    elif any(kw in lower for kw in ["thank you", "thanks", "appreciate"]):
        return random.choice([
            "You're very welcome! 💙",
            "Happy to help! 😊",
            "Anytime! Let me know if there's anything else."
        ])
    elif "your name" in lower:
        return "I'm CUAB Buddy, your friendly Crescent University assistant! (But you can call me Buddy 😉)"
    elif any(kw in lower for kw in ["who are you", "what are you"]):
        return "I'm your virtual assistant for Crescent University Abeokuta. I can help with admissions, courses, fees, and more!"
    return None

def detect_department(query):
    """Identify if the query mentions a specific department"""
    query = query.lower()
    for dept, aliases in DEPARTMENT_ALIASES.items():
        if any(alias in query for alias in [dept] + aliases):
            return dept
    return None

def extract_course_codes(text):
    """Extract and standardize course codes from text"""
    matches = re.findall(r'\b([a-z]{2,4}\s?-?\s?\d{3})\b', text.lower())
    return [match.replace(" ", "").replace("-", "") for match in matches]

def format_response(response, emotion, is_follow_up=False):
    """Make responses more natural and emotionally appropriate"""
    # Add emotional tone
    if emotion == "negative":
        response = f"I'm really sorry to hear that. 😔 Let me help - {response}"
    elif emotion == "positive":
        response = f"Great! 😊 {response}"
    
    # Add follow-up if appropriate
    if is_follow_up and st.session_state.last_topic:
        follow_ups = {
            "admission": "\n\nWould you like details about required documents or deadlines?",
            "fees": "\n\nNeed help with payment methods or installment plans?",
            "courses": "\n\nWant to check related courses or lecturers?",
            "hostel": "\n\nShould I explain the hostel application process?"
        }
        response += follow_ups.get(st.session_state.last_topic, "\n\nIs there anything else I can help with?")
    
    # Make responses more conversational
    response = response.replace("The answer is", "Here's what I found")
    response = response.replace("According to our records", "From what I know")
    
    return response

def search_answer(user_query, threshold=0.5):
    """Enhanced semantic search with context awareness"""
    try:
        # Normalize and process query
        processed_query = normalize_text(user_query)
        department = detect_department(processed_query)
        course_codes = extract_course_codes(processed_query)
        
        # Check for exact matches
        for qa in qa_data:
            normalized_db_q = normalize_text(qa["question"])
            
            # Direct match
            if processed_query == normalized_db_q:
                return qa["answer"]
            
            # Department context match
            if department and department in normalized_db_q:
                if processed_query in normalized_db_q:
                    return qa["answer"]
            
            # Course code match
            if course_codes and any(code in normalized_db_q for code in course_codes):
                if processed_query in normalized_db_q:
                    return qa["answer"]
        
        # Semantic search with context boosting
        query_embed = model.encode(processed_query, convert_to_tensor=True)
        scores = util.cos_sim(query_embed, qa_embeddings)[0]
        
        # Boost scores for relevant context
        for i, qa in enumerate(qa_data):
            normalized_db_q = normalize_text(qa["question"])
            if department and department in normalized_db_q:
                scores[i] += 0.2
            if course_codes and any(code in normalized_db_q for code in course_codes):
                scores[i] += 0.1
        
        best_idx = torch.argmax(scores).item()
        if scores[best_idx] > threshold:
            return qa_data[best_idx]["answer"]
        
        return None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def get_gpt_answer(user_query):
    """Get response from GPT with conversation context"""
    try:
        # Prepare conversation history
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[{"role": "user" if i % 2 == 0 else "assistant", "content": msg["user"] if i % 2 == 0 else msg["bot"]} 
              for i, msg in enumerate(st.session_state.chat_history[-4:])],
            {"role": "user", "content": user_query}
        ]
        
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            presence_penalty=0.5,
            frequency_penalty=0.5
        )
        
        answer = response.choices[0].message.content
        
        # Make it more conversational
        answer = answer.replace("However,", "That said,")
        answer = answer.replace("Furthermore,", "Also,")
        answer = re.sub(r"^Yes,", "Yeah,", answer)
        
        return answer
    except Exception as e:
        return random.choice([
            "Hmm, I'm having some trouble connecting. Could you ask again?",
            "Oops! My wires got crossed. Mind repeating that?"
        ])

def update_conversation_context(user_query, response):
    """Update the ongoing conversation context"""
    # Detect and store department
    department = detect_department(user_query)
    if department:
        st.session_state.conversation_context["current_department"] = department
    
    # Detect topic based on keywords
    query_lower = user_query.lower()
    if any(word in query_lower for word in ["admission", "apply", "requirement"]):
        st.session_state.last_topic = "admission"
    elif any(word in query_lower for word in ["fee", "payment", "tuition"]):
        st.session_state.last_topic = "fees"
    elif any(word in query_lower for word in ["course", "subject", "curriculum"]):
        st.session_state.last_topic = "courses"
    elif any(word in query_lower for word in ["hostel", "accommodation", "housing"]):
        st.session_state.last_topic = "hostel"
    
    # Update user mood
    st.session_state.conversation_context["user_mood"] = detect_emotion(user_query)

def stream_response(response):
    """Simulate typing for more natural interaction"""
    message_placeholder = st.empty()
    full_response = ""
    
    # Split into parts for more natural streaming
    parts = re.split(r'(?<=[,.!?])\s+', response)
    
    for part in parts:
        full_response += part + " "
        time.sleep(0.05 * len(part.split()))  # Vary delay based on part length
        message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)
    return full_response

def get_response(user_query):
    """Main function to generate appropriate response"""
    # Check for greetings
    if is_greeting(user_query):
        return get_greeting_response()
    
    # Handle small talk
    if small_talk_response := handle_small_talk(user_query):
        return small_talk_response
    
    # Check for follow-up questions
    is_follow_up = any(word in user_query.lower() for word in ["what about", "how about", "and"])
    
    # Try to find answer in QA database first
    qa_answer = search_answer(user_query)
    if qa_answer:
        response = qa_answer
    else:
        # Fall back to GPT
        response = get_gpt_answer(user_query)
    
    # Update conversation context
    update_conversation_context(user_query, response)
    
    # Format response naturally
    emotion = st.session_state.conversation_context["user_mood"]
    return format_response(response, emotion, is_follow_up)

# --- Main UI ---
def main():
    st.set_page_config(page_title="CUAB Buddy", page_icon="🎓", layout="centered")
    st.title("🎓 CUAB Buddy - Crescent University Assistant")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["bot"])
    
    # Chat input
    if user_input := st.chat_input("Ask me anything about Crescent University..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = get_response(user_input)
            full_response = stream_response(response)
        
        # Store in history
        st.session_state.chat_history.append({"user": user_input, "bot": full_response})

if __name__ == "__main__":
    main()
