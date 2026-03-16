# pip install yt-dlp
import yt_dlp

# Your target playlist URL
playlist_url = 'https://www.youtube.com/playlist?list=PLVItHqpXY_DC2tDsXAql81QjjgZj_wAPR'

ydl_opts = {
    'extract_flat': True,       # Only extracts metadata, doesn't download the videos
    'playlist_items': '1-200',  # Grabs exactly the first 200 videos in the playlist
    'quiet': True
}

print("Extracting up to 200 links from the playlist...")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(playlist_url, download=False)
    
    # Write the links to a text file
    with open('youtube_links.txt', 'w') as f:
        # Playlists store their videos in the 'entries' list
        if 'entries' in info:
            for entry in info['entries']:
                # Safety check: bypass deleted or private videos in the playlist
                if entry: 
                    link = entry.get('url')
                    f.write(f"{link}\n")

print("Done! Links saved to youtube_links.txt")
