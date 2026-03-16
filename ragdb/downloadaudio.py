import yt_dlp
import os
from pathlib import Path
import json

class PlaylistAudioDownloader:
    """
    Downloads audio from YouTube videos and saves them in the audiodata folder.
    """
    
    def __init__(self, output_dir="audiodata"):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store metadata for each video
        self.metadata_file = self.output_dir / "metadata.json"
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """Load existing metadata if available."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, indent=2, fp=f)
    
    def read_urls_from_file(self, file_path):
        """
        Read YouTube URLs from a text file.
        
        Args:
            file_path: Path to text file containing URLs (one per line)
        
        Returns:
            List of URLs
        """
        urls = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('https://'):
                    urls.append(line)
        return urls
    
    def download_from_url_list(self, urls, audio_format='mp3'):
        """
        Download audio from a list of YouTube video URLs.
        
        Args:
            urls: List of YouTube video URLs
            audio_format: Audio format to save (default: mp3)
        
        Returns:
            List of downloaded file paths
        """
        print(f"📥 Starting download of {len(urls)} videos")
        
        downloaded_files = []
        failed_downloads = []
        
        for idx, video_url in enumerate(urls, 1):
            print(f"\n[{idx}/{len(urls)}] Processing: {video_url}")
            
            try:
                result = self.download_single_video(video_url, audio_format, show_progress=False)
                if result:
                    downloaded_files.append(result)
                else:
                    failed_downloads.append(video_url)
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                failed_downloads.append(video_url)
                continue
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 Successfully downloaded: {len(downloaded_files)}/{len(urls)} videos")
        print(f"📁 Files saved in: {self.output_dir.absolute()}")
        
        if failed_downloads:
            print(f"\n❌ Failed downloads ({len(failed_downloads)}):")
            for url in failed_downloads:
                print(f"  - {url}")
        
        return downloaded_files
    
    def download_from_file(self, file_path, audio_format='mp3'):
        """
        Download audio from all URLs in a text file.
        
        Args:
            file_path: Path to text file containing URLs
            audio_format: Audio format to save (default: mp3)
        
        Returns:
            List of downloaded file paths
        """
        print(f"📄 Reading URLs from: {file_path}")
        urls = self.read_urls_from_file(file_path)
        
        if not urls:
            print("❌ No valid URLs found in file")
            return []
        
        return self.download_from_url_list(urls, audio_format)
    
    def download_single_video(self, video_url, audio_format='mp3', show_progress=True):
        """
        Download audio from a single YouTube video.
        
        Args:
            video_url: YouTube video URL
            audio_format: Audio format to save (default: mp3)
            show_progress: Whether to show detailed progress
        
        Returns:
            Path to downloaded file or None
        """
        if show_progress:
            print(f"📥 Downloading: {video_url}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': audio_format,
                'preferredquality': '192',
            }],
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'quiet': not show_progress,
            'no_warnings': not show_progress,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                safe_title = ydl.prepare_filename(info).replace('.webm', '').replace('.m4a', '')
                audio_file = f"{safe_title}.{audio_format}"
                
                # Store metadata
                video_id = info.get('id')
                self.metadata[video_id] = {
                    'title': info.get('title'),
                    'url': video_url,
                    'duration': info.get('duration'),
                    'audio_file': audio_file,
                }
                self.save_metadata()
                
                print(f"✅ Downloaded: {info.get('title')}")
                return audio_file
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def list_downloaded_files(self):
        """List all downloaded audio files."""
        audio_files = list(self.output_dir.glob("*.mp3"))
        print(f"\n📁 Found {len(audio_files)} audio files:")
        for file in audio_files:
            print(f"  - {file.name}")
        return audio_files


def main():
    """
    Main function to run the downloader.
    """
    # Initialize downloader
    downloader = PlaylistAudioDownloader(output_dir="audiodata")
    
    # Download from your youtube_links.txt file
    # file_path = "../yt-links/youtube_links.txt"
    # downloader.download_from_file(file_path)
    
    # Or download from a specific URL
    downloader.download_single_video("https://www.youtube.com/watch?v=fs5Idcn-8b0")
    
    # List all downloaded files
    # downloader.list_downloaded_files()


if __name__ == "__main__":
    main()