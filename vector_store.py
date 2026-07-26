import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# Cache of loaded models keyed by model_name, so multiple SimpleVectorStore
# instances sharing the same model_name don't each load their own copy of
# the transformer into memory.
_MODEL_CACHE = {}
_RERANKER_CACHE = {}
_warned_no_bm25 = False


def _get_model(model_name):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _get_reranker(model_name):
    # Lazy-loaded and cached: only paid for if a caller actually asks for
    # rerank=True, since this is a second model on top of the embedder.
    if model_name not in _RERANKER_CACHE:
        from sentence_transformers import CrossEncoder
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def _tokenize(text):
    return text.lower().split()


class SimpleVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = _get_model(model_name)
        self.documents = []
        self.metadatas = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self._bm25 = None  # built lazily, invalidated whenever add() is called

    # --- keep the (large, non-picklable-well) model out of the pickle file ---
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model", None)
        state.pop("_bm25", None)  # rebuilt on load; cheap and avoids stale index
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        model_name = getattr(self, "model_name", "all-MiniLM-L6-v2")
        self.model_name = model_name
        self.model = _get_model(model_name)
        self._bm25 = None

    def _embed(self, texts):
        # normalize_embeddings=True makes the dot product below equivalent
        # to cosine similarity. Without this, similarity scores are skewed
        # by document length/magnitude rather than semantic closeness.
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def _ensure_bm25(self):
        global _warned_no_bm25
        if not _BM25_AVAILABLE:
            if not _warned_no_bm25:
                print("⚠️ rank_bm25 not installed — falling back to semantic-only "
                      "search. Run: pip install rank_bm25")
                _warned_no_bm25 = True
            return None
        if self._bm25 is None and self.documents:
            self._bm25 = BM25Okapi([_tokenize(d) for d in self.documents])
        return self._bm25

    def add(self, documents, metadatas):
        if not documents:
            return
        emb = self._embed(documents).astype(np.float32)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        if self.embeddings.size == 0:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self._bm25 = None  # corpus changed, rebuild BM25 index on next query

    def query(
        self,
        query_text,
        n_results=3,
        filters=None,
        min_score=None,
        use_hybrid=True,
        hybrid_alpha=0.6,
        rerank=False,
        rerank_candidates=15,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        filters: optional dict of metadata field -> required value
                 (or a set/list of acceptable values), applied before ranking.
                 e.g. {"class": "10", "subject": "science"}
        min_score: optional float cutoff (0-1) on the *final* score used to
                   rank. Results below this are dropped rather than padded in.
        use_hybrid: combine semantic (embedding) similarity with BM25 keyword
                    matching. This matters a lot for exam prep — a student
                    asking about "Ohm's law" wants the chunk with that exact
                    term, not just something semantically nearby about
                    resistivity. Falls back to semantic-only if rank_bm25
                    isn't installed.
        hybrid_alpha: weight on the semantic score vs BM25 score, 0-1.
                      0.6 means 60% semantic, 40% keyword.
        rerank: if True, retrieves a wider candidate pool (rerank_candidates)
                using the fast hybrid score, then reorders the top slice with
                a cross-encoder for higher precision before cutting to
                n_results. Slower (loads/runs a second model) — worth it for
                the questions students actually depend on getting right, but
                you can leave it off for lower-stakes/high-volume calls.
        """
        if self.embeddings.size == 0:
            return {"documents": [[]], "metadatas": [[]], "scores": [[]]}

        candidate_idx = np.arange(len(self.documents))
        if filters:
            mask = []
            for i in candidate_idx:
                meta = self.metadatas[i]
                ok = True
                for key, wanted in filters.items():
                    val = meta.get(key)
                    if val is None:
                        continue  # metadata doesn't have this field; don't exclude
                    if isinstance(wanted, (list, set, tuple)):
                        if val not in wanted:
                            ok = False
                            break
                    elif val != wanted:
                        ok = False
                        break
                mask.append(ok)
            candidate_idx = candidate_idx[np.array(mask, dtype=bool)]

        if len(candidate_idx) == 0:
            return {"documents": [[]], "metadatas": [[]], "scores": [[]]}

        q_emb = self._embed([query_text])[0]
        sub_embeddings = self.embeddings[candidate_idx]
        sem_sim = np.dot(sub_embeddings, q_emb)  # already in [-1, 1], normalized vectors

        combined = sem_sim
        if use_hybrid:
            bm25 = self._ensure_bm25()
            if bm25 is not None:
                bm25_scores_full = bm25.get_scores(_tokenize(query_text))
                bm25_sub = bm25_scores_full[candidate_idx]
                # min-max normalize BM25 scores into [0, 1] so they're on a
                # comparable scale to cosine similarity before blending
                span = bm25_sub.max() - bm25_sub.min()
                bm25_norm = (bm25_sub - bm25_sub.min()) / span if span > 0 else np.zeros_like(bm25_sub)
                # sem_sim is roughly [0,1] for relevant text (normalized vectors),
                # blend directly
                combined = hybrid_alpha * sem_sim + (1 - hybrid_alpha) * bm25_norm

        pool_size = rerank_candidates if rerank else n_results
        order = np.argsort(combined)[::-1][:pool_size]
        pool_idx = candidate_idx[order]
        pool_scores = combined[order]

        if rerank and len(pool_idx) > 0:
            try:
                reranker = _get_reranker(reranker_model)
                pairs = [[query_text, self.documents[i]] for i in pool_idx]
                rerank_scores = reranker.predict(pairs)
                order2 = np.argsort(rerank_scores)[::-1][:n_results]
                top_idx = pool_idx[order2]
                # cross-encoder scores aren't bounded like cosine sim; keep
                # them as-is for ordering but don't apply min_score against
                # them since the scale differs from the hybrid score
                top_scores = np.array(rerank_scores)[order2]
            except Exception as e:
                print(f"⚠️ Reranker failed ({e}); falling back to hybrid ranking.")
                top_idx = pool_idx[:n_results]
                top_scores = pool_scores[:n_results]
        else:
            top_idx = pool_idx[:n_results]
            top_scores = pool_scores[:n_results]

        if min_score is not None and not rerank:
            keep = top_scores >= min_score
            top_idx = top_idx[keep]
            top_scores = top_scores[keep]

        return {
            "documents": [[self.documents[i] for i in top_idx]],
            "metadatas": [[self.metadatas[i] for i in top_idx]],
            "scores": [[float(s) for s in top_scores]],
        }
