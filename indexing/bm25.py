"""BM25 keyword index for sparse search leg of hybrid retrieval."""
import math
from collections import Counter


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.avg_dl = 0.0
        self.N = 0
        self.term_freqs = []
        self.vocab = set()

    def fit(self, keyword_texts: list):
        self.N = len(keyword_texts)
        self.term_freqs = []
        self.doc_lens = []

        for text in keyword_texts:
            tokens = text.lower().split()
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            for term in set(tokens):
                self.doc_freqs[term] += 1
                self.vocab.add(term)

        self.avg_dl = sum(self.doc_lens) / max(self.N, 1)

    def score(self, query: str, doc_idx: int) -> float:
        tokens = query.lower().split()
        tf = self.term_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0

        for term in tokens:
            if term not in tf:
                continue
            n = self.doc_freqs.get(term, 0)
            idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
            freq = tf[term]
            tf_norm = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
            score += idf * tf_norm

        return score

    def search(self, query: str, top_k: int = 20) -> list:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:top_k] if s > 0]
