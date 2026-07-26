import os
import pickle
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re
from dotenv import load_dotenv

load_dotenv()

# Jina AI API configuration
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v2-base-code"

class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.tokenized_docs = []
        
    def _get_embedding(self, text):
        if not JINA_API_KEY:
            raise ValueError("JINA_API_KEY not set. Please set it in your environment.")
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": JINA_MODEL,
            "input": text
        }
        try:
            response = requests.post(JINA_EMBEDDING_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            print(f"⚠️ Jina API error: {e}")
            return np.random.randn(768).tolist()
    
    def add(self, documents, metadatas):
        if not documents:
            return
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        for doc in documents:
            embedding = self._get_embedding(doc)
            self.embeddings.append(embedding)
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
    
    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())
    
    def _hybrid_score(self, query, n_results, min_score, rerank):
        query_embedding = self._get_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        tokenized_query = self._tokenize(query)
        bm25 = BM25Okapi(self.tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)
        max_sim = np.max(similarities) if np.max(similarities) > 0 else 1
        max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1
        norm_sim = similarities / max_sim
        norm_bm25 = bm25_scores / max_bm25
        hybrid_scores = 0.7 * norm_sim + 0.3 * norm_bm25
        indices = np.argsort(hybrid_scores)[::-1][:n_results]
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        for idx in indices:
            score = hybrid_scores[idx]
            if score >= min_score:
                results["documents"][0].append(self.documents[idx])
                results["metadatas"][0].append(self.metadatas[idx])
                results["distances"][0].append(1 - score)
        return results
    
    def query(self, query_text, n_results=5, filters=None, min_score=0.3, rerank=False, use_hybrid=True):
        if not self.documents:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        if filters:
            filtered_indices = []
            for i, meta in enumerate(self.metadatas):
                match = True
                for key, value in filters.items():
                    if meta.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(i)
            if not filtered_indices:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            temp_store = SimpleVectorStore()
            for idx in filtered_indices:
                temp_store.documents.append(self.documents[idx])
                temp_store.metadatas.append(self.metadatas[idx])
                temp_store.embeddings.append(self.embeddings[idx])
            temp_store.tokenized_docs = [self._tokenize(doc) for doc in temp_store.documents]
            return temp_store.query(query_text, n_results, min_score=min_score, rerank=rerank, use_hybrid=use_hybrid)
        if use_hybrid:
            return self._hybrid_score(query_text, n_results, min_score, rerank)
        query_embedding = self._get_embedding(query_text)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        indices = np.argsort(similarities)[::-1][:n_results]
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        for idx in indices:
            score = similarities[idx]
            if score >= min_score:
                results["documents"][0].append(self.documents[idx])
                results["metadatas"][0].append(self.metadatas[idx])
                results["distances"][0].append(1 - score)
        return results

def cosine_similarity_matrix(a, b):
    return cosine_similarity(a, b)
