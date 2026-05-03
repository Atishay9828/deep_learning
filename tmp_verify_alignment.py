import cv2
from pathlib import Path
import os
import pandas as pd
import numpy as np

video_root = Path("d:/NeuroBioSense_DL_Proj/NeuroBioSense/Advertisement Categories")
videos = sorted(list(video_root.rglob("*.mp4")))

total_samples = 0
emotions = []

print(f"Checking {len(videos)} videos...")

# CSV emotion sequence to compare
csv_path = "d:/NeuroBioSense_DL_Proj/NeuroBioSense/Biosignal Files/Pre-Processed/32-Hertz.csv"
df = pd.read_csv(csv_path, usecols=['EMOTION'])

for v in videos:
    cap = cv2.VideoCapture(str(v))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        duration_sec = frames / fps
        expected_samples = int(round(duration_sec * 32.0))
        total_samples += expected_samples
        
        # Get emotion from file path
        # Format: .../<ad_code>/<participant_id>/<emotion>/<filename>.mp4
        emotion = v.parent.name
        emotions.extend([emotion] * expected_samples)

print(f"Total summed samples based on video durations: {total_samples}")
print(f"Total samples in 32-Hertz.csv: {len(df)}")

if total_samples > 0:
    min_len = min(total_samples, len(df))
    # compare
    match = (np.array(emotions[:min_len]) == df['EMOTION'].values[:min_len]).mean()
    print(f"Emotion column match accuracy: {match*100:.2f}%")
