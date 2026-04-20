"""Cross-encoder reranker for precision scoring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RERANKER_MODEL, RERANKER_WEIGHT


class Reranker:
    def __init__(self, w_ce: float = RERANKER_WEIGHT):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(RERANKER_MODEL)
        self.w_ce = w_ce

    def rerank(self, query: str, items: list) -> list:
        if not items:
            return []

        pairs = [(query, it.get("text", "")) for it in items]
        ce_scores = self.model.predict(pairs).tolist()

        ret_scores = [it.get("score", 0.0) for it in items]
        mn, mx = min(ret_scores), max(ret_scores)
        if mx > mn:
            norm_ret = [(s - mn) / (mx - mn) for s in ret_scores]
        else:
            norm_ret = [0.5] * len(ret_scores)

        for it, s_ce, s_ret in zip(items, ce_scores, norm_ret):
            it["rerank_score"] = float(self.w_ce * s_ce + (1 - self.w_ce) * s_ret)

        return sorted(items, key=lambda x: x["rerank_score"], reverse=True)
