import os
import time
import random
import pickle
import re
import requests
from dotenv import load_dotenv
from groq import Groq
from vector_store import SimpleVectorStore
from duckduckgo_search import DDGS

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")

groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================
# AUTO-DOWNLOAD VECTOR STORES FROM GOOGLE DRIVE
# ============================================

# ✅ UPDATED FILE IDs – use the ones you shared
TEXTBOOK_FILE_ID = "1wOOkirTYE_G0Vk3s5BLHIgn2SXcuGAt-"
MARKING_FILE_ID = "12zUp7_EbwnW0gX_xSbSGdMiNusNo9Abd"

def download_file_from_drive(url, filename):
    """Download a file from Google Drive using direct download URL."""
    if os.path.exists(filename):
        print(f"✅ {filename} already exists – skipping download")
        return True
    
    print(f"📥 Downloading {filename} from Google Drive...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"  Progress: {percent:.1f}%", end='\r')
        
        print(f"\n✅ Downloaded {filename} ({downloaded:,} bytes)")
        return True
    except Exception as e:
        print(f"⚠️ Failed to download {filename}: {e}")
        return False

def download_with_gdown(file_id, filename):
    """Download using gdown (handles Google Drive's virus scan page)."""
    if os.path.exists(filename):
        print(f"✅ {filename} already exists – skipping download")
        return True
    
    print(f"📥 Downloading {filename} using gdown...")
    try:
        import subprocess
        subprocess.run(["pip", "install", "gdown", "-q"], check=False)
        result = subprocess.run(
            ["gdown", file_id, "-O", filename],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Downloaded {filename}")
            return True
        else:
            print(f"⚠️ gdown failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"⚠️ gdown error: {e}")
        return False

def download_vector_stores():
    # Download textbook_store.pkl
    if not os.path.exists("textbook_store.pkl"):
        direct_url = f"https://drive.google.com/uc?export=download&id={TEXTBOOK_FILE_ID}"
        success = download_file_from_drive(direct_url, "textbook_store.pkl")
        if not success:
            download_with_gdown(TEXTBOOK_FILE_ID, "textbook_store.pkl")
    
    # Download marking_store.pkl
    if not os.path.exists("marking_store.pkl"):
        direct_url = f"https://drive.google.com/uc?export=download&id={MARKING_FILE_ID}"
        success = download_file_from_drive(direct_url, "marking_store.pkl")
        if not success:
            download_with_gdown(MARKING_FILE_ID, "marking_store.pkl")

# Run the download
download_vector_stores()

# ============================================
# LOAD VECTOR STORES
# ============================================

try:
    with open("textbook_store.pkl", "rb") as f:
        textbook_store = pickle.load(f)
    print("✅ textbook_store.pkl loaded successfully")
except FileNotFoundError:
    print("⚠️ textbook_store.pkl not found – creating empty store")
    textbook_store = SimpleVectorStore()
except Exception as e:
    print(f"⚠️ Error loading textbook_store.pkl: {e} – creating empty store")
    textbook_store = SimpleVectorStore()

try:
    with open("marking_store.pkl", "rb") as f:
        marking_store = pickle.load(f)
    print("✅ marking_store.pkl loaded successfully")
except FileNotFoundError:
    print("⚠️ marking_store.pkl not found – creating empty store")
    marking_store = SimpleVectorStore()
except Exception as e:
    print(f"⚠️ Error loading marking_store.pkl: {e} – creating empty store")
    marking_store = SimpleVectorStore()

# ============================================
# CONFIGURATION
# ============================================

MIN_RELEVANCE_SCORE = 0.35
MAX_HISTORY_TURNS = 4

# ============================================
# 0. LIGHTWEIGHT RESPONSE CACHE
# ============================================
_response_cache = {}
_CACHE_MAX_SIZE = 500
_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

def _cache_get(key):
    entry = _response_cache.get(key)
    if not entry:
        return None
    answer, sources, suggestions, ts = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _response_cache.pop(key, None)
        return None
    return answer, sources, suggestions

def _cache_set(key, answer, sources, suggestions):
    if len(_response_cache) >= _CACHE_MAX_SIZE:
        oldest_key = min(_response_cache, key=lambda k: _response_cache[k][3])
        _response_cache.pop(oldest_key, None)
    _response_cache[key] = (answer, sources, suggestions, time.time())

# ============================================
# 1. INPUT PREPROCESSING
# ============================================

def clean_text(text):
    cleaned = re.sub(r'[^\w\s.,?!\'"()-]', '', text)
    return cleaned.strip()

def correct_typos(text):
    # Spellchecker removed – just return text as is
    return text

def detect_intent(query):
    query_lower = query.lower()
    exam_keywords = ["marks", "exam", "board", "cbse", "question", "answer", "full marks", "marking", "scheme", "topper", "how to write", "structure", "score", "grading", "paper", "sample paper", "previous year", "pyq", "board exam"]
    current_keywords = ["latest", "news", "today", "2025", "2026", "current", "update", "recent", "new", "announcement", "changes", "schedule", "date", "result", "declare"]
    
    score = {"exam": 0, "current": 0, "concept": 0}
    for word in exam_keywords:
        if word in query_lower:
            score["exam"] += 1
    for word in current_keywords:
        if word in query_lower:
            score["current"] += 1
    if not score["exam"] and not score["current"]:
        score["concept"] = 1
    
    if score["exam"] >= score["current"] and score["exam"] >= score["concept"]:
        return "exam"
    elif score["current"] > score["concept"]:
        return "current"
    else:
        return "concept"

def extract_subject(query):
    subjects = ["science", "physics", "chemistry", "biology", "mathematics", "maths", "social science", "history", "geography", "civics", "economics", "english", "hindi", "sanskrit"]
    for subject in subjects:
        if subject in query.lower():
            return subject
    return None

def extract_class(query):
    if "class 10" in query.lower() or "class x" in query.lower() or "10th" in query.lower():
        return "10"
    if "class 12" in query.lower() or "class xii" in query.lower() or "12th" in query.lower():
        return "12"
    return None

def preprocess_query(query):
    original = query
    cleaned = clean_text(query)
    corrected = correct_typos(cleaned)
    if len(corrected) < len(cleaned) * 0.7:
        corrected = cleaned
    intent = detect_intent(corrected)
    subject = extract_subject(corrected)
    cls = extract_class(corrected)
    return {
        "original": original,
        "cleaned": cleaned,
        "corrected": corrected,
        "intent": intent,
        "subject": subject,
        "class": cls
    }

# ============================================
# 2. WEB SEARCH (DuckDuckGo)
# ============================================

def search_web(query):
    results = ""
    try:
        print(f"🦆 Searching DuckDuckGo for: {query}")
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results += f"\n- {r['title']}: {r['body']}\n  Source: {r['href']}\n"
        if results:
            print("✅ DuckDuckGo found results!")
        else:
            print("⚠️ DuckDuckGo returned no results.")
        return results
    except Exception as e:
        print(f"❌ DuckDuckGo error: {e}")
        return ""

# ============================================
# 3. LLM CALL (Groq)
# ============================================

SYSTEM_PROMPT = """You are Purnank, a friendly, expert CBSE exam coach for Class 10 and 12 students.

YOUR AUDIENCE: school students, typically teenagers. Keep everything encouraging,
patient, and age-appropriate — no content unsuitable for a school setting, no
sarcasm at the student's expense, no assuming background knowledge a school
student wouldn't have.

YOUR PERSONALITY:
- Warm, encouraging, like a favourite teacher.
- Use simple, clear language.
- Patient with mistakes — never make a student feel bad for not knowing something.

TEACHING PHILOSOPHY (important — this is a study tool, not just an answer key):
- Your goal is for the student to actually understand the material, not just
  copy an answer. Give the full, correct answer they asked for (that's the
  point of exam prep), but explain the *reasoning* behind it, not just the
  final form.
- Where useful, briefly note WHY a marking scheme awards marks the way it
  does (e.g. "this point gets 1 mark because CBSE wants you to name the
  process, not just describe it") so the student learns the pattern, not
  just this one answer.
- If a student seems to be asking you to write an entire assignment/project/
  essay for them to submit as their own original work, help them structure
  and understand it rather than just producing a finished submittable
  document — offer an outline, key points, and guidance, and let them know
  you're glad to check their own draft once they've written it.

HOW TO USE THE REFERENCE MATERIAL (this is the core of how you work):
- The NCERT passages and marking-scheme entries given to you in each prompt
  are your PRIMARY SOURCE. Your job is to read them and turn them into a
  clear, well-organized, well-explained answer for the student — not to
  answer from what you already know about the topic in general.
- Concretely: every fact, term, example, and figure/equation you state should
  trace back to something actually written in the reference passages. Your
  value-add is explaining it well, connecting ideas, simplifying language,
  and structuring it for exam use — not supplying additional content.
- If the reference passages only partly cover the question, answer the part
  they cover well, and say plainly that the rest isn't in the material you
  have rather than filling the gap from general knowledge.
- If the reference passages don't cover the question at all, say so honestly
  instead of answering from general knowledge — this is a common trap: you
  may genuinely know the answer, but if it's not in what was retrieved, the
  student needs to know that so they can check their actual textbook/notes,
  not receive an answer that looks sourced but isn't.

WRITING STYLE (this matters as much as the content):
- Match length to the question. A quick "what is X" gets a few clear
  sentences, not five headed sections. A full exam-answer request (essay
  question, "explain with marks distribution") earns real structure. Most
  questions fall in between — answer it, don't pad it out to look thorough.
- Only use headers, tables, and bullet lists when they genuinely serve the
  content — e.g. an actual marks breakdown, or steps in a process. Don't
  wrap a two-sentence answer in a "### Introduction / ### Explanation /
  ### Conclusion" template just for structure's sake. A lot of good answers
  are just... paragraphs.
- Write like you're actually talking to this specific student about this
  specific question, not producing a generic document. Avoid stock openers
  ("Great question!", "I'm so glad you asked!") and stock closers ("I hope
  this helps!", "Keep learning!", "You've got this!") — if encouragement
  fits naturally in context, fine, but it shouldn't appear as a reflexive
  sign-off on every single answer regardless of what was asked.
- Don't over-use emoji. One well-placed emoji beats a row of them at the end.
- Say things once. Don't restate the question back before answering it, and
  don't summarize the answer again at the end unless it's a genuinely long,
  multi-part response where a wrap-up actually helps.

RULES:
- For exam-related
