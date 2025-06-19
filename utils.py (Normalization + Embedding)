import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from symspellpy import SymSpell, Verbosity
import pkg_resources

# Abbreviations & Synonyms
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

# Normalize input
def normalize_text(text):
    text = text.lower()
    for k, v in ABBREVIATIONS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    for k, v in SYNONYMS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text

# Spell correct
def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

def correct_text(sym_spell, input_text):
    suggestions = sym_spell.lookup_compound(input_text, max_edit_distance=2)
    return suggestions[0].term if suggestions else input_text

# Load dataset + embed questions
def load_data_and_embed(model, path="qa_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return data, embeddings
