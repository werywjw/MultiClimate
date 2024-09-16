from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import yt_dlp
import os
import subprocess

video_id = "Ry-ei9Bu8UI" # change here
new_filename = "test" # Change here


def split_transcript(video_id, interval=5):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    segments = []
    current_segment = ""
    current_time = 0
    next_time = interval
    
    for entry in transcript:
        start = entry['start']
        text = entry['text']
        if start >= next_time:
            segments.append({'time': current_time, 'text': current_segment})
            current_time = next_time
            next_time += interval
            current_segment = ""
        
        current_segment += " " + text
    
    segments.append({'time': current_time, 'text': current_segment.strip()})
    return segments

segments = split_transcript(video_id)


filename = f"dataset/{new_filename}/{new_filename}.csv"
def export_to_csv(segments, filename): 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(segments)
    df.to_csv(filename, index=False)

export_to_csv(segments, filename)


def download_youtube_video(url, output_path, new_filename):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_path, new_filename + '.%(ext)s'),
        'merge_output_format': 'mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

url = f"https://www.youtube.com/watch?v={video_id}"
output_path = f"dataset/{new_filename}/"

os.makedirs(output_path, exist_ok=True)

video_filename = download_youtube_video(url, output_path, new_filename)
print(f"Video downloaded to: {video_filename}")


input_video = f"dataset/{new_filename}/{new_filename}.mp4"
output_parent_folder = f"dataset/{new_filename}"
output_folder = os.path.join(output_parent_folder, f"{new_filename}_frames")

os.makedirs(output_folder, exist_ok=True)

ffmpeg_command = [
    "ffmpeg",
    "-i", input_video,
    "-r", "0.2",
    "-start_number", "0",
    os.path.join(output_folder, f"{new_filename}-%03d.jpg")
]

subprocess.run(ffmpeg_command, capture_output=True, text=True)

frame_to_delete = os.path.join(output_folder, f"{new_filename}-000.jpg")
os.remove(frame_to_delete)
