from typing import Dict, List
import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepeval.test_case import LLMTestCase


class LocalMetrics:
    """Local evaluation metrics without external API calls."""

    def __init__(self, embed_model: str = "nomic-embed-text"):
        self.embed_model = embed_model
        print(f"🔄 Using local Ollama embeddings for evaluation: {self.embed_model}")
        print("✅ Local evaluation model loaded\n")

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            # Older Ollama python clients
            resp = ollama.embeddings(model=self.embed_model, prompt=text)
            vectors.append(resp["embedding"])
        return np.array(vectors, dtype=np.float32)

    def semantic_similarity_score(self, actual: str, expected: str) -> float:
        actual_embedding = self._embed_texts([actual])
        expected_embedding = self._embed_texts([expected])
        similarity = cosine_similarity(actual_embedding, expected_embedding)[0][0]
        return float(similarity)

    def context_relevance_score(self, question: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        question_embedding = self._embed_texts([question])
        context_embeddings = self._embed_texts(contexts)
        similarities = cosine_similarity(question_embedding, context_embeddings)[0]
        return float(np.mean(similarities))

    def answer_relevance_score(self, question: str, answer: str) -> float:
        question_embedding = self._embed_texts([question])
        answer_embedding = self._embed_texts([answer])
        similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
        return float(similarity)

    def evaluate_test_case(self, test_case: LLMTestCase) -> Dict:
        return {
            "semantic_similarity": self.semantic_similarity_score(
                test_case.actual_output,
                test_case.expected_output
            ),
            "context_relevance": self.context_relevance_score(
                test_case.input,
                test_case.retrieval_context or []
            ),
            "answer_relevance": self.answer_relevance_score(
                test_case.input,
                test_case.actual_output
            )
        }


def initialize_metrics():
    return LocalMetrics(embed_model="nomic-embed-text")