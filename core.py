import os, time, random, pickle, re
from dotenv import load_dotenv
from groq import Groq
from vector_store import SimpleVectorStore
from duckduckgo_search import DDGS
from spellchecker import SpellChecker

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")

groq_client = Groq(api_key=GROQ_API_KEY)

# Load vector stores
with open("textbook_store.pkl", "rb") as f:
    textbook_store = pickle.load(f)
with open("marking_store.pkl", "rb") as f:
    marking_store = pickle.load(f)

_spell = SpellChecker()
MIN_RELEVANCE_SCORE = 0.35
MAX_HISTORY_TURNS = 4

# ============================================
# CACHE
# ============================================
_response_cache = {}
_CACHE_MAX_SIZE = 500
_CACHE_TTL_SECONDS = 6 * 60 * 60

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
# PREPROCESSING
# ============================================
def clean_text(text):
    cleaned = re.sub(r'[^\w\s.,?!\'"()-]', '', text)
    return cleaned.strip()

def correct_typos(text):
    words = text.split()
    corrected_words = []
    for word in words:
        if len(word) <= 2 or word.lower() in _spell:
            corrected_words.append(word)
        else:
            corrected = _spell.correction(word)
            corrected_words.append(corrected if corrected else word)
    return " ".join(corrected_words)

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
# WEB SEARCH
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
# LLM CALLS
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
- For exam-related questions: show marks distribution and cite sources (NCERT, Marking Scheme, or Web).
- For conceptual questions: explain clearly with examples; you can mention how it's tested but don't force a marks table unless asked.
- For current events: use web results and cite sources.
- If you don't know, say so and suggest where to look. Never invent facts,
  page numbers, or marks allocations that aren't in the reference material.
- CRITICAL — STAY WITHIN THE REFERENCE MATERIAL'S LEVEL: only use
  terminology, concepts, and detail that actually appear in the REFERENCE
  MATERIAL section of the prompt. You may know more advanced content from
  your training (e.g. Calvin cycle, Rubisco, and other Class 11/12-level
  biology terms when the reference material only covers a basic Class 10
  explanation of photosynthesis) — do NOT add it. CBSE syllabus is
  deliberately scoped by class/level, and a student who writes
  above-syllabus terminology in an exam can lose marks for going off the
  expected answer, not gain them. If the reference material is simple,
  your answer should be too, even if you personally know more.
- Understand typos and messy input.
- Stay within CBSE academic subjects (Class 10/12 curriculum) and study-skills
  topics (exam strategy, time management, stress before exams, etc). If asked
  something well outside that scope, gently redirect back to what you can help
  with rather than answering at length.
- If a student expresses serious stress, anxiety, or pressure about exams,
  respond with genuine care first before returning to the academic content —
  don't just plough ahead with the study answer.

Always cite your sources (NCERT, Marking Scheme, or Web) when you use them."""

DEFAULT_MODEL = "openai/gpt-oss-120b"
ENSEMBLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]
JUDGE_MODEL = "openai/gpt-oss-120b"

def call_llm(prompt, history=None, retries=3, model=DEFAULT_MODEL):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            if 'rate limit' in str(e).lower() or '429' in str(e):
                wait = (2 ** attempt) + random.random()
                print(f"⏳ Rate limit. Retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded.")

def call_llm_ensemble(prompt, history=None):
    import concurrent.futures

    def _call_one(model):
        try:
            return model, call_llm(prompt, history=history, retries=1, model=model)
        except Exception as e:
            print(f"⚠️ Ensemble member {model} failed: {e}")
            return model, None

    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ENSEMBLE_MODELS)) as executor:
        for model, answer in executor.map(_call_one, ENSEMBLE_MODELS):
            if answer:
                candidates.append((model, answer))

    if not candidates:
        raise Exception("All ensemble models failed.")
    if len(candidates) == 1:
        return candidates[0][1]

    labeled = "\n\n".join(
        f"--- ANSWER {chr(65+i)} (model: {m}) ---\n{a}"
        for i, (m, a) in enumerate(candidates)
    )

    judge_prompt = f"""You are judging {len(candidates)} candidate answers written by
different AI models for the same CBSE student question. Pick the single best
one, or synthesize a better answer that combines their strongest parts, using
these criteria in order of importance:

1. FAITHFULNESS: does it stick strictly to the reference material given below,
   with no invented facts, page numbers, or marks allocations? This includes
   terminology and concepts — if a candidate answer introduces terms or
   detail that go beyond what's in the reference material (even if factually
   correct in general, e.g. bringing in Class 11/12-level terminology to
   answer a Class 10-level reference), that counts AGAINST it. Prefer the
   candidate that stays closest to the actual scope of the reference material,
   not the one that sounds most comprehensive.
2. EXAM-CORRECTNESS: if this is an exam/marks question, is the marks
   distribution and structure accurate and CBSE-appropriate?
3. CLARITY: would a Class 10/12 student actually understand this?
4. TEACHING VALUE: does it help the student understand *why*, not just *what*?

ORIGINAL PROMPT GIVEN TO EACH MODEL:
{prompt}

{labeled}

Respond with ONLY the final best answer text — no meta-commentary about which
answer you picked or why, no "Answer A was better because...". The student
should never see this judging process, just the resulting answer. If you
synthesize from multiple candidates, remove any terminology or content that
isn't actually grounded in the reference material given above, even if it
appeared in one of the candidate answers."""

    return call_llm(judge_prompt, history=None, retries=2, model=JUDGE_MODEL)

def call_llm_stream(prompt, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": prompt})

    stream = groq_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1500,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

# ============================================
# CORE RESPONSE ENGINE
# ============================================

def build_context(query_info, n=3):
    corrected = query_info["corrected"]
    intent = query_info["intent"]
    subject = query_info["subject"]
    cls = query_info["class"]
    context = ""
    sources = []

    filters = {}
    if subject:
        filters["subject"] = subject
    if cls:
        filters["class"] = cls

    tb = textbook_store.query(corrected, n + 2, filters=filters or None, min_score=MIN_RELEVANCE_SCORE, rerank=True)
    mk = marking_store.query(corrected, n, filters=filters or None, min_score=MIN_RELEVANCE_SCORE, rerank=True)

    if filters and not tb['documents'][0]:
        tb = textbook_store.query(corrected, n + 2, min_score=MIN_RELEVANCE_SCORE, rerank=True)
    if filters and not mk['documents'][0]:
        mk = marking_store.query(corrected, n, min_score=MIN_RELEVANCE_SCORE, rerank=True)

    if tb['documents'][0]:
        context += "\n📖 From NCERT Textbooks (this is your primary source — build your answer from this):\n"
        for i, doc in enumerate(tb['documents'][0]):
            meta = tb['metadatas'][0][i]
            loc = f" (p.{meta['page']})" if meta.get('page') else ""
            context += f"\n  --- Passage {i+1}: [{meta.get('source', 'NCERT')}{loc}] ---\n  {doc}\n"
            sources.append({
                "type": "textbook",
                "source": meta.get('source', 'NCERT'),
                "page": meta.get('page'),
                "subject": meta.get('subject'),
                "class": meta.get('class'),
                "snippet": doc[:350] + ('...' if len(doc) > 350 else '')
            })

    if mk['documents'][0]:
        context += "\n📝 From CBSE Marking Schemes:\n"
        for i, doc in enumerate(mk['documents'][0]):
            ans = mk['metadatas'][0][i].get('answer', 'N/A')
            context += f"  Q: {doc}\n  ✅ Answer: {ans}\n"
            sources.append({
                "type": "marking_scheme",
                "source": "CBSE Marking Scheme",
                "page": None,
                "subject": mk['metadatas'][0][i].get('subject'),
                "class": mk['metadatas'][0][i].get('class'),
                "snippet": f"Q: {doc[:150]}...\nA: {ans[:200]}..."
            })

    if intent == "current" or (not tb['documents'][0] and not mk['documents'][0]):
        web_results = search_web(corrected)
        if web_results:
            context += f"\n🌐 From Internet Search:\n{web_results}"
            sources.append({
                "type": "web",
                "source": "DuckDuckGo Search",
                "page": None,
                "subject": None,
                "class": None,
                "snippet": web_results[:300] + ('...' if len(web_results) > 300 else '')
            })

    if not context:
        context = "\n⚠️ No relevant information found in NCERT, marking schemes, or web search. Please try rephrasing your question."

    return context, sources

def get_purnank_response(query, n=3, history=None, use_ensemble=False):
    query_info = preprocess_query(query)
    print(f"\n📝 Query Info:")
    print(f"  Original: {query_info['original']}")
    print(f"  Corrected: {query_info['corrected']}")
    print(f"  Intent: {query_info['intent']}")
    print(f"  Subject: {query_info['subject']}")
    print(f"  Class: {query_info['class']}")

    corrected = query_info["corrected"]
    intent = query_info["intent"]
    subject = query_info["subject"]
    cls = query_info["class"]

    cache_key = None
    if not history:
        cache_key = (corrected.lower().strip(), subject, cls, intent, n, use_ensemble)
        cached = _cache_get(cache_key)
        if cached is not None:
            print("⚡ Served from cache")
            return cached

    context, sources = build_context(query_info, n)

    subject_line = ""
    if subject:
        subject_line += f" (Subject: {subject})"
    if cls:
        subject_line += f" (Class: {cls})"
    
    prompt = f"""You are Purnank, a CBSE exam coach.{subject_line}

INTENT DETECTED: {intent.upper()}

REFERENCE MATERIAL:
{context}

USER'S QUESTION: {corrected}

INSTRUCTIONS:
"""
    
    if intent == "exam":
        prompt += """
- This is an EXAM question. Show a clear marks distribution (e.g., "2 marks for definition, 3 marks for explanation...") and cite your sources.
- Scale the structure to the actual size of the answer: a 1-2 mark question needs a couple of clear sentences, not full headed sections. A 5+ mark question earns real structure (e.g. a short intro, the marked points, maybe a diagram/equation if relevant). Don't force "Introduction / Key Points / Conclusion" headers onto a short answer just for the sake of a template.
"""
    elif intent == "current":
        prompt += """
- This is a CURRENT EVENTS question. Use the web sources provided and cite them clearly.
- Focus on factual, up-to-date information.
"""
    else:
        prompt += """
- This is a CONCEPTUAL question. Explain the concept clearly with examples.
- You may mention how this is tested in exams, but do not force a marks table unless the user specifically asks for it.
- Keep the tone friendly and easy to understand.
"""

    prompt += """
Always:
- Cite your sources (NCERT, Marking Scheme, or Web).
- If you don't know, say so and suggest where to look.
- Answer like you're actually talking to this student, not filling out a template. No stock opener, no stock "hope this helps" closer — just answer well and stop.
- IMPORTANT: After your main answer, add a section titled '**Follow-up questions:**' (exactly, with the bold markdown). Then list 3 short, relevant follow-up questions as separate bullet points (using '- '). Do not include any other text after this list. These will be extracted and shown to the student as clickable suggestions.
"""
    
    raw_answer = call_llm_ensemble(prompt, history=history) if use_ensemble else call_llm(prompt, history=history)

    # --- Parse suggestions ---
    suggestions = []
    clean_answer = raw_answer
    if '**Follow-up questions:**' in raw_answer:
        parts = raw_answer.split('**Follow-up questions:**', 1)
        clean_answer = parts[0].strip()
        raw_suggestions = parts[1].strip()
        for line in raw_suggestions.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                suggestions.append(line[2:].strip())
            elif line.startswith('* '):
                suggestions.append(line[2:].strip())
            elif line.startswith('> '):
                suggestions.append(line[2:].strip())
        suggestions = suggestions[:3]
    
    # Fallback suggestions if the model didn't generate any
    if not suggestions:
        suggestions = [
            "Explain this in simpler terms",
            "Give me an example",
            "How is this tested in exams?"
        ]

    if cache_key is not None:
        _cache_set(cache_key, clean_answer, sources, suggestions)

    return clean_answer, sources, suggestions

def get_purnank_response_stream(query, n=3, history=None):
    query_info = preprocess_query(query)
    corrected = query_info["corrected"]
    intent = query_info["intent"]
    subject = query_info["subject"]
    cls = query_info["class"]

    context, _ = build_context(query_info, n)

    subject_line = ""
    if subject:
        subject_line += f" (Subject: {subject})"
    if cls:
        subject_line += f" (Class: {cls})"

    prompt = f"""You are Purnank, a CBSE exam coach.{subject_line}

INTENT DETECTED: {intent.upper()}

REFERENCE MATERIAL:
{context}

USER'S QUESTION: {corrected}

INSTRUCTIONS:
"""
    if intent == "exam":
        prompt += """
- This is an EXAM question. Show a clear marks distribution (e.g., "2 marks for definition, 3 marks for explanation...") and cite your sources.
- Scale the structure to the actual size of the answer: a 1-2 mark question needs a couple of clear sentences, not full headed sections. A 5+ mark question earns real structure (e.g. a short intro, the marked points, maybe a diagram/equation if relevant). Don't force "Introduction / Key Points / Conclusion" headers onto a short answer just for the sake of a template.
"""
    elif intent == "current":
        prompt += """
- This is a CURRENT EVENTS question. Use the web sources provided and cite them clearly.
- Focus on factual, up-to-date information.
"""
    else:
        prompt += """
- This is a CONCEPTUAL question. Explain the concept clearly with examples.
- You may mention how this is tested in exams, but do not force a marks table unless the user specifically asks for it.
- Keep the tone friendly and easy to understand.
"""
    prompt += """
Always:
- Cite your sources (NCERT, Marking Scheme, or Web).
- If you don't know, say so and suggest where to look.
- Answer like you're actually talking to this student, not filling out a template. No stock opener, no stock "hope this helps" closer — just answer well and stop."""

    yield from call_llm_stream(prompt, history=history)
