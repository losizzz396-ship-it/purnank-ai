import os, json, pickle, re
from tqdm import tqdm
from vector_store import SimpleVectorStore
import pdfplumber  # More robust PDF parser

# Expected layout (recommended):
#   ./data/textbooks/10/science.pdf
#   ./data/textbooks/12/biology.pdf
# If your PDFs aren't organized into class subfolders yet, this will fall
# back to guessing class/subject from the filename (e.g. "10_science.pdf"),
# and leave class=None if it truly can't tell — better than silently
# tagging everything wrong.
TEXTBOOK_DIR = "./data/textbooks"
MARKING_FILE = "./data/marking/output_clean.json"

CLASS_PATTERN = re.compile(r'(?:class[\s_-]?)?(10|12)\b', re.IGNORECASE)


def guess_class_subject(rel_path):
    """rel_path like '10/science.pdf' or '10_science.pdf' or 'science.pdf'."""
    parts = re.split(r'[\\/]', rel_path)
    cls = None
    for p in parts[:-1]:  # any parent folder named "10" or "12"
        if p in ("10", "12"):
            cls = p
            break
    filename = os.path.splitext(parts[-1])[0]
    if cls is None:
        m = CLASS_PATTERN.search(filename)
        if m:
            cls = m.group(1)
    subject = CLASS_PATTERN.sub('', filename).strip('_- ').lower() or filename.lower()
    return cls, subject


def chunk_text(text, size=500, overlap=50):
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    return [' '.join(words[i:i + size]) for i in range(0, len(words), step)]


def ingest_textbooks(store):
    pdf_paths = []
    for root, _, files in os.walk(TEXTBOOK_DIR):
        for f in files:
            if f.endswith('.pdf'):
                pdf_paths.append(os.path.relpath(os.path.join(root, f), TEXTBOOK_DIR))

    for rel_path in tqdm(pdf_paths, desc="Textbooks"):
        full_path = os.path.join(TEXTBOOK_DIR, rel_path)
        cls, subject = guess_class_subject(rel_path)
        if cls is None:
            print(f"⚠️ Could not determine class for {rel_path} — "
                  f"put it in a '10/' or '12/' subfolder for reliable filtering.")

        docs, metas = [], []
        try:
            with pdfplumber.open(full_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        continue
                    for i, chunk in enumerate(chunk_text(text)):
                        docs.append(chunk)
                        metas.append({
                            "source": rel_path,
                            "subject": subject,
                            "class": cls,
                            "page": page_num,
                            "chunk": i,
                        })
        except Exception as e:
            print(f"⚠️ Error reading {rel_path}: {e}")
            continue

        store.add(docs, metas)


def ingest_marking(store):
    with open(MARKING_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qa = data.get("qa_pairs", [])
    docs, metas = [], []
    for p in qa:
        docs.append(f"Q{p['question_number']} ({p['marks']} marks): {p['question_text']}")
        metas.append({
            "answer": p.get("model_answer", ""),
            "q_num": p["question_number"],
            "marks": p["marks"],
            "subject": p.get("subject"),   # only populated if present in your JSON
            "class": p.get("class"),       # only populated if present in your JSON
        })
    store.add(docs, metas)


if __name__ == "__main__":
    textbook_store = SimpleVectorStore()
    marking_store = SimpleVectorStore()
    ingest_textbooks(textbook_store)
    ingest_marking(marking_store)

    # Write to temp files first so a crash mid-write can't corrupt the
    # store you're currently serving in production.
    with open("textbook_store.pkl.tmp", "wb") as f:
        pickle.dump(textbook_store, f)
    with open("marking_store.pkl.tmp", "wb") as f:
        pickle.dump(marking_store, f)
    os.replace("textbook_store.pkl.tmp", "textbook_store.pkl")
    os.replace("marking_store.pkl.tmp", "marking_store.pkl")

    print(f"✅ Ingestion complete – {len(textbook_store.documents)} textbook chunks, "
          f"{len(marking_store.documents)} marking-scheme entries saved.")
