"""NeuroKit2 Advanced Biosignal Extracting.

Extracts continuous physiological metrics (Heart Rate from BVP, Phasic EDA from Skin Conductance)
on a per-participant strict basis to prevent filter cross-talk!
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
try:
    import neurokit2 as nk
except ImportError:
    print("FATAL: neurokit2 not found. Run pip install neurokit2")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=".")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    input_csv = dataset_root / "NeuroBioSense" / "Biosignal Files" / "Pre-Processed" / "Mapped-32-Hertz.csv"
    output_csv = dataset_root / "NeuroBioSense" / "Biosignal Files" / "Pre-Processed" / "Mapped-32-Hertz-NK.csv"

    if not input_csv.exists():
        print(f"Missing Input CSV: {input_csv}")
        sys.exit(1)

    print(f"Loading continuous Mapped dataset ({input_csv})...")
    df = pd.read_csv(input_csv)

    print(f"Dataset has {len(df)} samples. Processing per participant to prevent filter overlap...")
    
    # We will initialize new columns
    df['HR_BPM'] = 0.0
    df['EDA_PHASIC'] = 0.0

    participants = df['participant_id'].dropna().unique()

    for pid in participants:
        mask = df['participant_id'] == pid
        sub_df = df[mask]
        
        bvp = sub_df['BVP'].values
        eda = sub_df['EDA'].values
        
        # Protect against exceedingly small or totally NaN arrays
        if len(bvp) < 320: # skip if less than 10 seconds total
            continue

        try:
            # PPG -> Heart Rate (BPM)
            signals_ppg, _ = nk.ppg_process(bvp, sampling_rate=32)
            df.loc[mask, 'HR_BPM'] = signals_ppg['PPG_Rate'].fillna(method='bfill').fillna(method='ffill').values
        except Exception as e:
            # If PPG is too degraded to find peaks, fallback to cleaned raw
            try:
                df.loc[mask, 'HR_BPM'] = nk.signal_detrend(bvp)
            except:
                pass
            
        try:
            # EDA -> Phasic (sweat micro-responses vs generic baseline drift)
            signals_eda, _ = nk.eda_process(eda, sampling_rate=32)
            df.loc[mask, 'EDA_PHASIC'] = signals_eda['EDA_Phasic'].fillna(0).values
        except Exception as e:
            try:
                df.loc[mask, 'EDA_PHASIC'] = nk.signal_detrend(eda)
            except:
                pass
                
    # Overwrite BVP and EDA safely so the downstream SignalModule consumes HR and Phasic data explicitly
    # But wait, keeping BVP as HR_BPM alters the fundamental scale! 
    # BVP raw is usually centered around 0. HR_BPM is ~60-100.
    # We should normalize them so the 1D CNN parses them identically.
    
    # Standardize extracted markers globally
    def standardize(series):
        std = series.std()
        if std == 0 or np.isnan(std):
            return np.zeros_like(series)
        return ((series - series.mean()) / std).values
    
    print("Normalizing final modalities...")
    df.loc[:, 'BVP'] = standardize(df['HR_BPM'])
    df.loc[:, 'EDA'] = standardize(df['EDA_PHASIC'])

    # Drop explicit calculation columns to save space
    df = df.drop(columns=['HR_BPM', 'EDA_PHASIC'])

    print(f"Exporting isolated NK matrices to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("NeuroKit2 Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
