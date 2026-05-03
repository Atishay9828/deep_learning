import pandas as pd
import numpy as np
import os
from pathlib import Path

csv_path = "d:/NeuroBioSense_DL_Proj/NeuroBioSense/Biosignal Files/Pre-Processed/32-Hertz.csv"
df = pd.read_csv(csv_path, usecols=['EMOTION'])

# count contiguous blocks of emotions
emotions = df['EMOTION'].values
shifts = emotions[:-1] != emotions[1:]
num_blocks = np.sum(shifts) + 1
print(f"Number of contiguous emotion blocks in CSV: {num_blocks}")

# get the sequence of blocks
block_indices = np.concatenate(([0], np.where(shifts)[0] + 1))
block_emotions = emotions[block_indices]

from collections import Counter
print("Block emotion counts:", Counter(block_emotions))

# Count video files
video_root = Path("d:/NeuroBioSense_DL_Proj/NeuroBioSense/Advertisement Categories")
all_vids = list(video_root.rglob("*.mp4"))
print(f"Total video files: {len(all_vids)}")

csv_em_counts = df['EMOTION'].value_counts()
print("\nCSV raw row counts per emotion:")
print(csv_em_counts)
