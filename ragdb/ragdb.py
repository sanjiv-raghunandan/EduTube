import json
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions


class RAGDatabase:
    """
    Generates embeddings from cleaned chunks and inserts them into ChromaDB.
    Reads directly from cleandata/ produced by cleantranscript.py.
    """

    def __init__(self, clean_dir="cleandata", db_dir="chromadb_store", collection_name="edutube"):
        """
        Args:
            clean_dir: Directory containing chunk JSON files from cleantranscript.py
            db_dir: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
        """
        self.clean_dir = Path(clean_dir)
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)

        # Load chunk metadata
        self.chunk_metadata_file = self.clean_dir / "chunk_metadata.json"
        self.chunk_metadata = self.load_chunk_metadata()

        # Track what's been inserted
        self.ingestion_log_file = self.db_dir / "ingestion_log.json"
        self.ingestion_log = self.load_ingestion_log()

        # -------------------------------------------------------
        # Embedding function
        # Uses nomic-embed-text via local Ollama engine
        # Massive 8192 token limit, state-of-the-art for RAG
        # -------------------------------------------------------
        print("🔄 Connecting to local Ollama for nomic-embed-text...")
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text"
        )
        print("✅ Embedding model connected")

        # -------------------------------------------------------
        # ChromaDB persistent client + collection
        # -------------------------------------------------------
        print(f"🔄 Connecting to ChromaDB at: {self.db_dir.absolute()}")
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
        )
        print(f"✅ Connected to collection '{collection_name}' "
              f"(current count: {self.collection.count()} documents)")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def load_chunk_metadata(self):
        if self.chunk_metadata_file.exists():
            with open(self.chunk_metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def load_ingestion_log(self):
        if self.ingestion_log_file.exists():
            with open(self.ingestion_log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_ingestion_log(self):
        with open(self.ingestion_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.ingestion_log, indent=2, fp=f)

    def load_chunk_file(self, chunk_id):
        """Load a single chunk JSON file from cleandata/."""
        chunk_file = self.clean_dir / f"{chunk_id}.json"
        if not chunk_file.exists():
            return None
        with open(chunk_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # INGESTION
    # ------------------------------------------------------------------

    def ingest_video_chunks(self, base_name, video_meta, batch_size=50):
        """
        Embed and insert all chunks for a single video into ChromaDB.

        Args:
            base_name: Key from chunk_metadata (transcript stem)
            video_meta: Metadata dict for this video from chunk_metadata.json
            batch_size: Number of chunks to upsert per ChromaDB call

        Returns:
            Number of chunks inserted, or 0 on failure
        """
        chunk_ids = video_meta.get('chunk_ids', [])
        if not chunk_ids:
            return 0

        documents = []
        metadatas = []
        ids = []

        for chunk_id in chunk_ids:
            chunk = self.load_chunk_file(chunk_id)
            if not chunk or not chunk.get('text', '').strip():
                continue

            documents.append(chunk['text'])
            ids.append(chunk['chunk_id'])
            metadatas.append({
                'video_id':          chunk.get('video_id', ''),
                'video_title':       chunk.get('video_title', 'Unknown'),
                'video_url':         chunk.get('video_url', ''),
                'source_transcript': chunk.get('source_transcript', ''),
                'chunk_index':       chunk.get('chunk_index', 0),
                'total_chunks':      chunk.get('total_chunks', 0),
                'word_count':        chunk.get('word_count', 0),
            })

        if not documents:
            return 0

        # Upsert in batches to avoid memory issues with large videos
        inserted = 0
        for i in range(0, len(documents), batch_size):
            batch_docs  = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids   = ids[i:i + batch_size]

            self.collection.upsert(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas,
            )
            inserted += len(batch_docs)

        return inserted

    def ingest_all(self, batch_size=50):
        """
        Ingest all chunks from cleandata/ into ChromaDB.
        Skips videos already present in the ingestion log.
        """
        if not self.chunk_metadata:
            print("❌ No chunk metadata found. Run cleantranscript.py first.")
            return

        total_videos = len(self.chunk_metadata)
        print(f"📊 Found {total_videos} videos to ingest\n")

        total_inserted = 0
        skipped = 0
        failed = 0

        for base_name, video_meta in tqdm(self.chunk_metadata.items(), desc="Ingesting"):
            # Resume support: skip already ingested videos
            if base_name in self.ingestion_log:
                skipped += 1
                continue

            try:
                count = self.ingest_video_chunks(base_name, video_meta, batch_size)

                if count > 0:
                    total_inserted += count
                    self.ingestion_log[base_name] = {
                        'video_title':   video_meta.get('video_title', 'Unknown'),
                        'chunks_inserted': count,
                    }
                    # Save log after each video so progress is not lost on interruption
                    self.save_ingestion_log()
                    print(f"✅ {video_meta.get('video_title', base_name)[:60]} → {count} chunks")
                else:
                    failed += 1
                    print(f"⚠️  No chunks inserted for: {base_name}")

            except Exception as e:
                failed += 1
                print(f"❌ Error ingesting {base_name}: {str(e)}")
                continue

        # Final summary
        print(f"\n{'='*60}")
        print(f"🎉 Ingestion Complete!")
        print(f"✅ Videos processed:  {total_videos - skipped - failed}")
        print(f"⏭️  Skipped (already done): {skipped}")
        print(f"❌ Failed:            {failed}")
        print(f"📦 Total chunks in DB: {self.collection.count()}")
        print(f"💾 ChromaDB stored at: {self.db_dir.absolute()}")

    # ------------------------------------------------------------------
    # QUERYING (used later by the Streamlit app / Llama model)
    # ------------------------------------------------------------------

    def query(self, query_text, n_results=5):
        """
        Semantic search over the ChromaDB collection.

        Args:
            query_text: The user's question / search string
            n_results: Number of top results to return

        Returns:
            List of dicts with 'text', 'video_title', 'video_url', 'score'
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        hits = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            hits.append({
                'text':        doc,
                'video_title': meta.get('video_title', 'Unknown'),
                'video_url':   meta.get('video_url', ''),
                'chunk_index': meta.get('chunk_index', 0),
                'score':       round(1 - (dist / 2), 4),  # cosine distance → similarity
            })

        return hits

    def print_query_results(self, query_text, n_results=5):
        """Helper to display query results in a readable format."""
        print(f"\n🔍 Query: \"{query_text}\"")
        print("=" * 60)

        hits = self.query(query_text, n_results)

        for i, hit in enumerate(hits, 1):
            print(f"\n[{i}] Score: {hit['score']}")
            print(f"    Video: {hit['video_title']}")
            print(f"    URL:   {hit['video_url']}")
            print(f"    Text:  {hit['text'][:200]}...")

        return hits

    def collection_stats(self):
        """Print stats about the current collection."""
        count = self.collection.count()
        print(f"\n📊 Collection Stats:")
        print(f"   Total documents (chunks): {count}")
        print(f"   Videos ingested:          {len(self.ingestion_log)}")
        print(f"   DB location:              {self.db_dir.absolute()}")


def main():
    """
    Main function to run ingestion.
    """
    db = RAGDatabase(
        clean_dir="cleandata",
        db_dir="chromadb_store",
        collection_name="edutube"
    )

    # Ingest all chunks into ChromaDB
    db.ingest_all(batch_size=50)

    # Print collection stats after ingestion
    db.collection_stats()

    # Test a query
    # db.print_query_results("what is gradient descent?", n_results=5)


if __name__ == "__main__":
    main()