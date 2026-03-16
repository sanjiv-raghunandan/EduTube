import re
import json
from pathlib import Path
from tqdm import tqdm


class TranscriptCleaner:
    """
    Cleans raw Whisper transcripts and generates text chunks for RAG.
    """

    def __init__(self, transcript_dir="transcriptdata", clean_dir="cleandata2"):
        """
        Args:
            transcript_dir: Directory containing raw transcripts
            clean_dir: Directory to save cleaned chunks
        """
        self.transcript_dir = Path(transcript_dir)
        self.clean_dir = Path(clean_dir)
        self.clean_dir.mkdir(exist_ok=True)

        # Load transcript metadata
        self.transcript_metadata_file = self.transcript_dir / "transcript_metadata.json"
        self.transcript_metadata = self.load_transcript_metadata()

        # Chunk metadata to pass to ragdb.py
        self.chunk_metadata_file = self.clean_dir / "chunk_metadata.json"
        self.chunk_metadata = self.load_chunk_metadata()

    def load_transcript_metadata(self):
        existing = {}
        if self.transcript_metadata_file.exists():
            with open(self.transcript_metadata_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)

        # Discover all .txt files on disk, regardless of metadata JSON
        for txt_file in self.transcript_dir.glob("*.txt"):
            base_name = txt_file.stem
            if base_name in existing:
                continue
            segments_file = self.transcript_dir / f"{base_name}_segments.json"
            existing[base_name] = {
                'audio_file':       f"{base_name}.mp3",
                'transcript_file':  txt_file.name,
                'segments_file':    segments_file.name if segments_file.exists() else '',
                'language':         'en',
                'video_id':         '',
                'video_title':      base_name,
                'video_url':        '',
                'duration':         0,
                'char_count':       0,
                'word_count':       0,
            }
        return existing

    def load_chunk_metadata(self):
        if self.chunk_metadata_file.exists():
            with open(self.chunk_metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_chunk_metadata(self):
        with open(self.chunk_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, indent=2, fp=f)

    # ------------------------------------------------------------------
    # CLEANING
    # ------------------------------------------------------------------

    def clean_text(self, text):
        """
        Clean raw Whisper transcript text safely for technical RAG.
        """
        # Safely target ONLY specific known Whisper audio tags, preserving code brackets
        whisper_tags = [
            r'\[Music\]', r'\[Silence\]', r'\[Applause\]', r'\[Laughter\]', 
            r'\(applause\)', r'\(laughs\)', r'\(sighs\)'
        ]
        for tag in whisper_tags:
            text = re.sub(tag, '', text, flags=re.IGNORECASE)

        # Only remove pure vocal stumbles. 
        # Kept OUT: right, like, actually, basically, you know
        fillers = [
            r'\bum+\b', r'\buh+\b', r'\bhmm+\b', 
            r'\bokay so\b', r'\bso yeah\b', r'\byeah so\b'
        ]
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)

        # Remove repeated punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s*,\s*,+', ',', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)

        # Collapse multiple spaces / newlines into a single space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    # ------------------------------------------------------------------
    # CHUNKING
    # ------------------------------------------------------------------

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """
        Split cleaned text into overlapping word-based chunks.

        Args:
            text: Cleaned text string
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between consecutive chunks

        Returns:
            List of chunk strings
        """
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)

            # If we've reached the end, stop
            if end >= len(words):
                break

            # Move forward by (chunk_size - overlap) words
            start += chunk_size - overlap

        return chunks

    # ------------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------------

    def process_transcript(self, base_name, video_metadata, chunk_size=500, overlap=50):
        """
        Clean and chunk a single transcript.

        Args:
            base_name: Stem of the transcript filename
            video_metadata: Metadata dict from transcript_metadata.json
            chunk_size: Words per chunk
            overlap: Overlapping words between chunks

        Returns:
            Number of chunks generated, or 0 on failure
        """
        transcript_file = self.transcript_dir / video_metadata['transcript_file']

        if not transcript_file.exists():
            print(f"❌ Transcript file not found: {transcript_file.name}")
            return 0

        # Read raw transcript
        with open(transcript_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        if not raw_text.strip():
            print(f"⚠️  Empty transcript: {transcript_file.name}")
            return 0

        # Clean
        cleaned_text = self.clean_text(raw_text)

        # Chunk
        chunks = self.chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)

        if not chunks:
            print(f"⚠️  No chunks generated for: {transcript_file.name}")
            return 0

        # Save each chunk as a separate JSON file (easy for ragdb.py to load)
        chunk_records = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{base_name}_chunk_{i:04d}"
            chunk_file = self.clean_dir / f"{chunk_id}.json"

            chunk_data = {
                'chunk_id': chunk_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'text': chunk_text,
                'word_count': len(chunk_text.split()),
                # Carry over video metadata for ChromaDB
                'video_id': video_metadata.get('video_id', ''),
                'video_title': video_metadata.get('video_title', 'Unknown'),
                'video_url': video_metadata.get('video_url', ''),
                'source_transcript': video_metadata['transcript_file'],
            }

            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, indent=2, fp=f)

            chunk_records.append(chunk_id)

        # Update chunk metadata
        self.chunk_metadata[base_name] = {
            'video_title': video_metadata.get('video_title', 'Unknown'),
            'video_url': video_metadata.get('video_url', ''),
            'video_id': video_metadata.get('video_id', ''),
            'transcript_file': video_metadata['transcript_file'],
            'total_chunks': len(chunks),
            'chunk_ids': chunk_records,
            'chunk_size': chunk_size,
            'overlap': overlap,
        }

        return len(chunks)

    def process_all_transcripts(self, chunk_size=500, overlap=50):
        """
        Clean and chunk all transcripts in transcriptdata/.

        Args:
            chunk_size: Words per chunk
            overlap: Overlapping words between chunks
        """
        if not self.transcript_metadata:
            print("❌ No transcript metadata found. Run extracttranscript.py first.")
            return

        total = len(self.transcript_metadata)
        print(f"📊 Found {total} transcripts to process")
        print(f"⚙️  Chunk size: {chunk_size} words | Overlap: {overlap} words\n")

        total_chunks = 0
        skipped = 0
        failed = 0

        for base_name, video_metadata in tqdm(self.transcript_metadata.items(), desc="Cleaning & Chunking"):
            # Skip if already processed
            if base_name in self.chunk_metadata:
                skipped += 1
                continue

            count = self.process_transcript(base_name, video_metadata, chunk_size, overlap)

            if count > 0:
                total_chunks += count
                print(f"✅ {video_metadata.get('video_title', base_name)[:60]} → {count} chunks")
            else:
                failed += 1

        # Save updated metadata
        self.save_chunk_metadata()

        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 Cleaning & Chunking Complete!")
        print(f"✅ Processed: {total - skipped - failed}")
        print(f"⏭️  Skipped (already done): {skipped}")
        print(f"❌ Failed: {failed}")
        print(f"📦 Total chunks generated: {total_chunks}")
        print(f"📁 Chunks saved in: {self.clean_dir.absolute()}")

    def list_chunks(self):
        """Print a summary of all chunk files."""
        chunk_files = list(self.clean_dir.glob("*.json"))
        chunk_files = [f for f in chunk_files if f.name != "chunk_metadata.json"]
        print(f"\n📁 Found {len(chunk_files)} chunk files in {self.clean_dir}")
        return chunk_files


def main():
    """
    Main function to run cleaning and chunking.
    """
    cleaner = TranscriptCleaner(
        transcript_dir="transcriptdata",
        clean_dir="cleandata2"
    )

    # Process all transcripts
    # chunk_size: words per chunk (500 is a good default for RAG)
    # overlap: shared words between adjacent chunks (prevents context loss at boundaries)
    cleaner.process_all_transcripts(chunk_size=200, overlap=40)

    # List all generated chunks
    # cleaner.list_chunks()


if __name__ == "__main__":
    main()