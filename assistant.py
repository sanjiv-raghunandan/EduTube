from pathlib import Path
from ragdb.ragdb import RAGDatabase
import ollama


class EduTubeAssistant:
    """
    Combines RAG retrieval with LLM inference for question answering.
    """

    def __init__(self, model_name="YOUR_CUSTOM_MODEL", db_dir="ragdb/chromadb_store", clean_dir="ragdb/cleandata"):
        self.model_name = model_name
        self.db = RAGDatabase(
            clean_dir=clean_dir,
            db_dir=db_dir,
            collection_name="edutube"
        )

    def query_with_context(self, question, n_results=5, temperature=0.7, max_tokens=2000):
        """
        Answer a question using RAG context.
        """
        results = self.db.query(question, n_results=n_results)

        context_parts = []
        for i, hit in enumerate(results, 1):
            context_parts.append(f"[Source {i}] {hit['text']}")
        context_str = "\n\n".join(context_parts)

        augmented_prompt = f"""Answer the following question based on the provided context from educational videos.

Context:
{context_str}

Question: {question}

Answer based on the context above. If the context doesn't contain relevant information, say so."""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an educational AI assistant specialized in programming and data structures."},
                {"role": "user", "content": augmented_prompt}
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        )

        return {
            "answer": response["message"]["content"],
            "sources": results,
            "context": context_str
        }

    def query_with_context_stream(self, question, n_results=5, temperature=0.7, max_tokens=2000):
        """
        Stream version for Streamlit.
        """
        results = self.db.query(question, n_results=n_results)

        context_parts = []
        for i, hit in enumerate(results, 1):
            context_parts.append(f"[Source {i}] {hit['text']}")
        context_str = "\n\n".join(context_parts)

        yield {"type": "context", "sources": results, "context": context_str}

        augmented_prompt = f"""Answer the following question based on the provided context from educational videos.

Context:
{context_str}

Question: {question}

Answer based on the context above. If the context doesn't contain relevant information, say so."""

        stream = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an educational AI assistant specialized in programming and data structures."},
                {"role": "user", "content": augmented_prompt}
            ],
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        )

        for chunk in stream:
            if chunk["message"]["content"]:
                yield {"type": "chunk", "text": chunk["message"]["content"]}