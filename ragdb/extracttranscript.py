import whisper
import os
from pathlib import Path
import json
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class TranscriptExtractor:
    """
    Extracts transcripts from audio files using OpenAI Whisper base model.
    """
    
    def __init__(self, audio_dir="audiodata", transcript_dir="transcriptdata"):
        """
        Initialize the transcript extractor.
        
        Args:
            audio_dir: Directory containing audio files
            transcript_dir: Directory to save transcript files
        """
        self.audio_dir = Path(audio_dir)
        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(exist_ok=True)
        
        # Load audio metadata
        self.audio_metadata_file = self.audio_dir / "metadata.json"
        self.audio_metadata = self.load_audio_metadata()
        
        # Load Whisper model
        print("🔄 Loading Whisper base model...")
        self.model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
    
    def load_audio_metadata(self):
        """Load audio metadata from audiodata folder."""
        if self.audio_metadata_file.exists():
            with open(self.audio_metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def transcribe_audio(self, audio_file_path):
        """
        Transcribe a single audio file using Whisper.
        
        Args:
            audio_file_path: Path to audio file
        
        Returns:
            Dictionary containing transcript and metadata
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                str(audio_file_path),
                language="en",  # Change if needed, or set to None for auto-detect
                verbose=False
            )
            
            return {
                'text': result['text'],
                'language': result['language'],
                'segments': result['segments'],  # Timestamped segments
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            return {
                'text': None,
                'error': str(e),
                'success': False
            }
    
    def extract_all_transcripts(self):
        """
        Extract transcripts from all audio files in the audiodata folder.
        
        Returns:
            Dictionary with success/failure counts
        """
        # Get all audio files
        audio_files = list(self.audio_dir.glob("*.mp3")) + \
                     list(self.audio_dir.glob("*.m4a")) + \
                     list(self.audio_dir.glob("*.webm"))
        
        if not audio_files:
            print("❌ No audio files found in audiodata folder")
            return {'success': 0, 'failed': 0}
        
        print(f"📊 Found {len(audio_files)} audio files to transcribe")
        print(f"🔄 Starting transcription using Whisper base model...\n")
        
        success_count = 0
        failed_count = 0
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc="Transcribing"):
            audio_filename = audio_file.name
            base_name = audio_file.stem
            
            # Check if already transcribed
            transcript_file = self.transcript_dir / f"{base_name}.txt"
            if transcript_file.exists():
                print(f"⏭️  Skipping (already transcribed): {audio_filename}")
                continue
            
            print(f"\n🎤 Transcribing: {audio_filename}")
            
            # Transcribe
            result = self.transcribe_audio(audio_file)
            
            if result['success']:
                # Save transcript text
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                
                # Save detailed transcript with timestamps (optional)
                segments_file = self.transcript_dir / f"{base_name}_segments.json"
                with open(segments_file, 'w', encoding='utf-8') as f:
                    json.dump(result['segments'], indent=2, fp=f)
                
                # Find matching video metadata from audio metadata
                video_id = None
                video_info = {}
                for vid_id, metadata in self.audio_metadata.items():
                    if metadata.get('audio_file') and base_name in metadata['audio_file']:
                        video_id = vid_id
                        video_info = metadata
                        break
                
                success_count += 1
                print(f"✅ Transcribed: {audio_filename}")
                print(f"   Words: {len(result['text'].split())}, Language: {result['language']}")
                
            else:
                failed_count += 1
                print(f"❌ Failed: {audio_filename}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 Transcription Complete!")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📁 Transcripts saved in: {self.transcript_dir.absolute()}")
        
        return {'success': success_count, 'failed': failed_count}
    
    def transcribe_single_file(self, audio_filename):
        """
        Transcribe a specific audio file.
        
        Args:
            audio_filename: Name of the audio file in audiodata folder
        
        Returns:
            Transcript text or None
        """
        audio_file = self.audio_dir / audio_filename
        
        if not audio_file.exists():
            print(f"❌ Audio file not found: {audio_filename}")
            return None
        
        print(f"🎤 Transcribing: {audio_filename}")
        result = self.transcribe_audio(audio_file)
        
        if result['success']:
            base_name = audio_file.stem
            transcript_file = self.transcript_dir / f"{base_name}.txt"
            
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            
            print(f"✅ Transcript saved: {transcript_file.name}")
            return result['text']
        
        return None
    
    def list_transcripts(self):
        """List all transcript files."""
        transcript_files = list(self.transcript_dir.glob("*.txt"))
        print(f"\n📁 Found {len(transcript_files)} transcript files:")
        for file in transcript_files:
            print(f"  - {file.name}")
        return transcript_files


def main():
    """
    Main function to run transcript extraction.
    """
    # Initialize extractor
    extractor = TranscriptExtractor(
        audio_dir="audiodata",
        transcript_dir="transcriptdata"
    )
    
    # Extract all transcripts
    extractor.extract_all_transcripts()
    
    # Or transcribe a single file
    # extractor.transcribe_single_file("your_audio_file.mp3")
    
    # List all transcripts
    # extractor.list_transcripts()


if __name__ == "__main__":
    main()