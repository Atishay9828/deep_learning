"""Script to reconstruct missing participant structure in the continuous 32-Hertz NeuroBioSense biosignal file.

It loops over the alphabetically sorted datasets and assigns metadata matching the physical clips.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=".")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    video_root = dataset_root / "NeuroBioSense" / "Advertisement Categories"
    input_csv = dataset_root / "NeuroBioSense" / "Biosignal Files" / "Pre-Processed" / "32-Hertz.csv"
    output_csv = dataset_root / "NeuroBioSense" / "Biosignal Files" / "Pre-Processed" / "Mapped-32-Hertz.csv"

    if not input_csv.exists():
        print(f"Cannot find input CSV: {input_csv}")
        sys.exit(1)

    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv)
    
    print(f"Scanning videos in {video_root} ...")
    videos = sorted(list(video_root.rglob("*.mp4")), key=lambda p: str(p).replace('\\', '/').lower())
    
    print(f"Found {len(videos)} clips. Reconstructing array mappings...")

    p_ids = []
    ad_codes = []
    video_names = []

    cumulative_samples = 0

    for v in videos:
        # Expected format: Advertisement Categories/<Category>/<ad_code>/<participant_id>/<emotion>/<filename>.video
        parts = list(v.relative_to(video_root).parts)
        if len(parts) < 4:
            continue
        
        ad_code = parts[-4]
        participant_id = parts[-3]
        video_name = v.name

        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps <= 0:
            print(f"WARNING: Bad FPS for {v}")
            continue
            
        duration_sec = frames / fps
        expected_samples = int(round(duration_sec * 32.0))
        
        p_ids.extend([participant_id] * expected_samples)
        ad_codes.extend([ad_code] * expected_samples)
        video_names.extend([video_name] * expected_samples)
        
        cumulative_samples += expected_samples

    print(f"Total reconstructed chunks: {cumulative_samples}")
    print(f"Actual dataset raw chunks : {len(df)}")
    
    # We truncate or pad to perfectly match the dataframe size
    min_len = min(cumulative_samples, len(df))
    
    df['participant_id'] = ""
    df['ad_code'] = ""
    df['video_name'] = ""
    
    df.loc[:min_len-1, 'participant_id'] = p_ids[:min_len]
    df.loc[:min_len-1, 'ad_code'] = ad_codes[:min_len]
    df.loc[:min_len-1, 'video_name'] = video_names[:min_len]
    
    print(f"Writing to {output_csv} ...")
    df.to_csv(output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
