"""Generates NeuroBioSense Kaggle notebook (.ipynb)."""
import json, textwrap
from pathlib import Path

def md(s):
    lines = textwrap.dedent(s).strip().split("\n")
    return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in lines[:-1]]+[lines[-1]]}

def code(s):
    lines = textwrap.dedent(s).strip().split("\n")
    return {"cell_type":"code","metadata":{},"outputs":[],"execution_count":None,
            "source":[l+"\n" for l in lines[:-1]]+[lines[-1]]}

cells = []

# ── 1. Title ──
cells.append(md("""\
    # 🧠 NeuroBioSense — Full T4 GPU Pipeline
    **Preprocessing → EDA → Training → Evaluation → Inference**
    
    Uses Kaggle dataset `alicia2uu/neurobiosense-dataset`"""))

# ── 2. Setup ──
cells.append(code("""\
    import os, sys, time, warnings, shutil
    warnings.filterwarnings('ignore')
    os.system('pip install -q facenet-pytorch neurokit2 seaborn openpyxl')
    import torch, numpy as np, pandas as pd
    import matplotlib.pyplot as plt, seaborn as sns
    import cv2
    from pathlib import Path
    from collections import Counter
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")"""))

# ── 3. Clone Repo ──
cells.append(code("""\
    REPO = Path('/kaggle/working/repo')
    if not REPO.exists():
        os.system('git clone https://github.com/HaryiankKumra/NeuroBioSense_DL_Proj.git '+str(REPO))
    sys.path.insert(0, str(REPO))
    os.chdir(str(REPO))
    print("✅ Repo ready")"""))

# ── 4. Dataset Discovery ──
cells.append(code("""\
    INPUT = Path('/kaggle/input/neurobiosense-dataset')
    WORK = Path('/kaggle/working')
    def find_file(root, name):
        for p in root.rglob(name): return p
        return None
    def find_dir(root, name):
        for p in root.rglob(name):
            if p.is_dir(): return p
        return None
    SIGNAL_CSV = find_file(INPUT, '32-Hertz.csv')
    VIDEO_ROOT = find_dir(INPUT, 'Advertisement Categories')
    DEMO_FILE = find_file(INPUT, 'Participant_demographic_information.xlsx')
    print(f"Signal CSV : {SIGNAL_CSV}")
    print(f"Video Root : {VIDEO_ROOT}")
    print(f"Demographics: {DEMO_FILE}")
    for p in sorted(INPUT.rglob('*')):
        d = len(p.relative_to(INPUT).parts)
        if d <= 2:
            pre = "  "*(d-1)
            if p.is_file(): print(f"{pre}📄 {p.name} ({p.stat().st_size/1e6:.1f}MB)")
            else: print(f"{pre}📁 {p.name}/")"""))

# ── 5. Biosignal EDA ──
cells.append(md("## 📊 Biosignal Exploration"))
cells.append(code("""\
    df_raw = pd.read_csv(SIGNAL_CSV)
    alias = {'X':'ACC_X','Y':'ACC_Y','Z':'ACC_Z'}
    df_raw = df_raw.rename(columns={k:v for k,v in alias.items() if k in df_raw.columns})
    print(f"Shape: {df_raw.shape}, Columns: {list(df_raw.columns)}")
    print(df_raw.describe())
    print("\\n--- Data Quality ---")
    for col in df_raw.columns:
        n = df_raw[col].isna().sum()
        if n > 0: print(f"  ⚠️ {col}: {n} NaN")
    sig_cols = [c for c in ['BVP','EDA','TEMP','ACC_X','ACC_Y','ACC_Z'] if c in df_raw.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, col in enumerate(sig_cols):
        ax = axes[i//3, i%3]
        v = df_raw[col].dropna().values
        ax.hist(v, bins=100, alpha=0.7, edgecolor='k', linewidth=0.3)
        ax.set_title(f'{col}\\nμ={np.mean(v):.2f} σ={np.std(v):.2f}')
        ax.axvline(np.mean(v), color='r', ls='--', alpha=0.7)
    plt.suptitle('Raw Biosignal Distributions (32Hz)', fontsize=16, weight='bold')
    plt.tight_layout(); plt.savefig(WORK/'biosignal_dist.png', dpi=150); plt.show()"""))

# ── 6. Emotion Balance ──
cells.append(code("""\
    emos = df_raw['EMOTION'].value_counts()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    emos.plot(kind='bar', ax=a1, color=sns.color_palette('husl', len(emos)), edgecolor='k')
    a1.set_title('Emotion Class Distribution'); a1.set_ylabel('Samples')
    emos.plot(kind='pie', ax=a2, autopct='%1.1f%%', colors=sns.color_palette('husl', len(emos)))
    a2.set_title('Emotion Proportions'); a2.set_ylabel('')
    plt.suptitle('Class Balance', fontsize=16, weight='bold')
    plt.tight_layout(); plt.savefig(WORK/'emotion_balance.png', dpi=150); plt.show()
    # Per-emotion signal overlay
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, col in enumerate(sig_cols):
        ax = axes[i//3, i%3]
        for em in sorted(df_raw['EMOTION'].unique()):
            ax.hist(df_raw[df_raw['EMOTION']==em][col].dropna(), bins=50, alpha=0.35, label=em, density=True)
        ax.set_title(f'{col} by Emotion'); ax.legend(fontsize=7)
    plt.suptitle('Per-Emotion Signal Distributions', fontsize=16, weight='bold')
    plt.tight_layout(); plt.savefig(WORK/'signals_per_emotion.png', dpi=150); plt.show()"""))

# ── 7. Video Analysis ──
cells.append(md("## 🎬 Video Analysis"))
cells.append(code("""\
    all_v = sorted(list(VIDEO_ROOT.rglob('*.mp4'))+list(VIDEO_ROOT.rglob('*.MP4')))
    seen=set(); videos=[]
    for v in all_v:
        k=str(v).lower()
        if k not in seen: seen.add(k); videos.append(v)
    print(f"Total clips: {len(videos)}")
    durs, fcs, cats, emos_v, pids_v = [], [], [], [], []
    for v in videos:
        cap=cv2.VideoCapture(str(v)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
        fc=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        durs.append(fc/fps); fcs.append(fc)
        parts=v.relative_to(VIDEO_ROOT).parts
        if len(parts)>=4: cats.append(parts[0]); emos_v.append(parts[-2]); pids_v.append(parts[-3])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].hist(durs, bins=40, color='steelblue', edgecolor='k')
    axes[0,0].set_title(f'Duration (μ={np.mean(durs):.1f}s)'); axes[0,0].set_xlabel('sec')
    axes[0,1].hist(fcs, bins=40, color='coral', edgecolor='k')
    axes[0,1].set_title(f'Frame Count (μ={np.mean(fcs):.0f})')
    cc=Counter(cats); axes[1,0].bar(cc.keys(), cc.values(), color=sns.color_palette('husl',len(cc)))
    axes[1,0].set_title('Videos per Category'); axes[1,0].tick_params(axis='x', rotation=20)
    ec=Counter(emos_v); axes[1,1].bar(ec.keys(), ec.values(), color=sns.color_palette('husl',len(ec)))
    axes[1,1].set_title('Videos per Emotion')
    plt.suptitle('Video Dataset Statistics', fontsize=16, weight='bold')
    plt.tight_layout(); plt.savefig(WORK/'video_stats.png', dpi=150); plt.show()
    print(f"Duration: {min(durs):.1f}s-{max(durs):.1f}s | Participants: {len(set(pids_v))}")"""))

# ── 8. Biosignal Reconstruction ──
cells.append(md("## 🔧 Biosignal Reconstruction → `Mapped-32-Hertz.csv`"))
cells.append(code("""\
    WORK_DS = WORK/'NeuroBioSense'
    BIO_OUT = WORK_DS/'Biosignal Files'/'Pre-Processed'
    BIO_OUT.mkdir(parents=True, exist_ok=True)
    # Symlink directories from input
    for d in INPUT.rglob('*'):
        if d.is_dir() and len(d.relative_to(INPUT).parts)==1:
            dst = WORK_DS/d.name
            if not dst.exists():
                try: os.symlink(str(d), str(dst))
                except: shutil.copytree(str(d), str(dst))
    shutil.copy2(str(SIGNAL_CSV), str(BIO_OUT/'32-Hertz.csv'))
    if DEMO_FILE:
        dd=WORK_DS/'Participant Data'; dd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(DEMO_FILE), str(dd/DEMO_FILE.name))
    # Reconstruct mapping
    df = pd.read_csv(BIO_OUT/'32-Hertz.csv')
    vr = WORK_DS/'Advertisement Categories'
    vs = sorted(list(vr.rglob('*.mp4'))+list(vr.rglob('*.MP4')), key=lambda p:str(p).replace('\\\\','/').lower())
    seen2=set(); uvs=[]
    for vv in vs:
        kk=str(vv).lower()
        if kk not in seen2: seen2.add(kk); uvs.append(vv)
    p_ids, ad_codes, vid_names, cum = [], [], [], 0
    for v in uvs:
        parts=list(v.relative_to(vr).parts)
        if len(parts)<4: continue
        cap=cv2.VideoCapture(str(v)); fps=cap.get(cv2.CAP_PROP_FPS); frames=cap.get(cv2.CAP_PROP_FRAME_COUNT); cap.release()
        if fps<=0: continue
        n=int(round((frames/fps)*32.0))
        p_ids.extend([parts[-3]]*n); ad_codes.extend([parts[-4]]*n); vid_names.extend([v.name]*n); cum+=n
    print(f"Reconstructed: {cum} | CSV rows: {len(df)}")
    ml=min(cum, len(df))
    df['participant_id']=''; df['ad_code']=''; df['video_name']=''
    df.loc[:ml-1,'participant_id']=p_ids[:ml]; df.loc[:ml-1,'ad_code']=ad_codes[:ml]; df.loc[:ml-1,'video_name']=vid_names[:ml]
    mapped_csv = BIO_OUT/'Mapped-32-Hertz.csv'
    df.to_csv(mapped_csv, index=False)
    print(f"✅ Saved {mapped_csv} ({mapped_csv.stat().st_size/1e6:.1f} MB)")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    df[df['participant_id']!='']['participant_id'].value_counts().head(20).plot(kind='bar',ax=a1,color='steelblue')
    a1.set_title('Top 20 Participants'); a1.tick_params(axis='x',rotation=45)
    df[df['ad_code']!='']['ad_code'].value_counts().head(20).plot(kind='bar',ax=a2,color='coral')
    a2.set_title('Top 20 Ad Codes'); a2.tick_params(axis='x',rotation=45)
    plt.suptitle('Mapping Validation', fontsize=16, weight='bold')
    plt.tight_layout(); plt.savefig(WORK/'mapping_validation.png', dpi=150); plt.show()"""))

# ── 9. NeuroKit2 Processing ──
cells.append(md("## 🧪 NeuroKit2 Biosignal Processing"))
cells.append(code("""\
    import neurokit2 as nk
    df = pd.read_csv(mapped_csv)
    df_raw_bvp = df['BVP'].copy(); df_raw_eda = df['EDA'].copy()
    df['HR_BPM']=0.0; df['EDA_PHASIC']=0.0
    participants = df['participant_id'].dropna().unique()
    participants = [p for p in participants if p != '']
    print(f"Processing {len(participants)} participants...")
    for i, pid in enumerate(participants):
        mask = df['participant_id']==pid
        bvp=df.loc[mask,'BVP'].values; eda=df.loc[mask,'EDA'].values
        if len(bvp)<320: continue
        try:
            sig,_=nk.ppg_process(bvp, sampling_rate=32)
            df.loc[mask,'HR_BPM']=sig['PPG_Rate'].fillna(method='bfill').fillna(method='ffill').values
        except:
            try: df.loc[mask,'HR_BPM']=nk.signal_detrend(bvp)
            except: pass
        try:
            sig,_=nk.eda_process(eda, sampling_rate=32)
            df.loc[mask,'EDA_PHASIC']=sig['EDA_Phasic'].fillna(0).values
        except:
            try: df.loc[mask,'EDA_PHASIC']=nk.signal_detrend(eda)
            except: pass
        if (i+1)%10==0: print(f"  {i+1}/{len(participants)}")
    def standardize(s):
        std=s.std()
        if std==0 or np.isnan(std): return np.zeros_like(s)
        return ((s-s.mean())/std).values
    df['BVP']=standardize(df['HR_BPM']); df['EDA']=standardize(df['EDA_PHASIC'])
    df=df.drop(columns=['HR_BPM','EDA_PHASIC'])
    nk_csv = BIO_OUT/'Mapped-32-Hertz-NK.csv'
    df.to_csv(nk_csv, index=False)
    print(f"✅ Saved {nk_csv} ({nk_csv.stat().st_size/1e6:.1f} MB)")"""))

# ── 10. Before/After Comparison ──
cells.append(code("""\
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    n_show = min(5000, len(df))
    axes[0,0].plot(df_raw_bvp.values[:n_show], alpha=0.7, linewidth=0.5)
    axes[0,0].set_title('Raw BVP (first 5000 samples)'); axes[0,0].set_ylabel('BVP')
    axes[0,1].plot(df['BVP'].values[:n_show], alpha=0.7, linewidth=0.5, color='green')
    axes[0,1].set_title('Processed BVP (HR-derived, standardized)'); axes[0,1].set_ylabel('BVP')
    axes[1,0].plot(df_raw_eda.values[:n_show], alpha=0.7, linewidth=0.5)
    axes[1,0].set_title('Raw EDA (first 5000 samples)'); axes[1,0].set_ylabel('EDA')
    axes[1,1].plot(df['EDA'].values[:n_show], alpha=0.7, linewidth=0.5, color='green')
    axes[1,1].set_title('Processed EDA (Phasic, standardized)'); axes[1,1].set_ylabel('EDA')
    plt.suptitle('NeuroKit2: Before vs After Processing', fontsize=16, weight='bold')
    plt.tight_layout(); plt.savefig(WORK/'neurokit_comparison.png', dpi=150); plt.show()
    # Correlation heatmap
    corr = df[sig_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
    ax.set_title('Post-Processing Signal Correlation', fontsize=14)
    plt.tight_layout(); plt.savefig(WORK/'signal_correlation.png', dpi=150); plt.show()"""))

# ── 11. Training ──
cells.append(md("## 🏋️ Model Training (T4 GPU)"))
cells.append(code("""\
    os.chdir(str(REPO))
    # Point training to our working dataset
    DS_ROOT = str(WORK)
    SIGNAL = str(BIO_OUT/'Mapped-32-Hertz.csv')
    VID = str(WORK_DS/'Advertisement Categories')
    OUTPUT = str(WORK/'multimodal_stage3.pth')
    cmd = (
        f"python -m emotion_recognition.scripts.train_multimodal "
        f"--dataset-root {DS_ROOT} "
        f"--video-root {VID} "
        f"--signal-csv {SIGNAL} "
        f"--device cuda "
        f"--epochs 20 "
        f"--batch-size 16 "
        f"--neuro-only "
        f"--task valence2 "
        f"--loss-type focal "
        f"--patience 8 "
        f"--num-workers 2 "
        f"--output {OUTPUT}"
    )
    print(f"Running: {cmd}")
    os.system(cmd)
    print("✅ Training complete!")"""))

# ── 12. Parse Training Log + Curves ──
cells.append(md("## 📈 Training Curves"))
cells.append(code("""\
    # If JSON report exists, load and plot
    report_path = WORK/'multimodal_stage3.json'
    if report_path.exists():
        import json
        with open(report_path) as f: report = json.load(f)
        print(json.dumps(report, indent=2))
        # Per-class accuracy
        if 'test_per_class_acc' in report and 'class_names' in report:
            fig, ax = plt.subplots(figsize=(10, 5))
            names = report['class_names']
            accs = report['test_per_class_acc']
            bars = ax.bar(names, accs, color=sns.color_palette('husl', len(names)), edgecolor='k')
            for b, a in zip(bars, accs): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{a:.2f}', ha='center')
            ax.set_title(f"Per-Class Test Accuracy | Overall: {report.get('test_overall_acc',0):.4f}", fontsize=14)
            ax.set_ylim(0, 1.1)
            plt.tight_layout(); plt.savefig(WORK/'per_class_acc.png', dpi=150); plt.show()
        # Confusion matrix
        if 'test_confusion_matrix' in report:
            cm = np.array(report['test_confusion_matrix'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=report.get('class_names',[]),
                        yticklabels=report.get('class_names',[]))
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title(f"Confusion Matrix | F1={report.get('test_macro_f1',0):.4f}", fontsize=14)
            plt.tight_layout(); plt.savefig(WORK/'confusion_matrix.png', dpi=150); plt.show()
    else:
        print("No training report found - check training output above for errors")"""))

# ── 13. Detailed Evaluation ──
cells.append(md("## 🔬 Detailed Evaluation"))
cells.append(code("""\
    ckpt_path = WORK/'multimodal_stage3.pth'
    if ckpt_path.exists():
        from emotion_recognition.models.full_model import MultimodalEmotionModel
        from emotion_recognition.utils.dataset import build_neurobiosense_datasets
        from emotion_recognition.utils.signal_processing import SignalNormalizationStats
        payload = torch.load(str(ckpt_path), map_location='cpu')
        stats_dict = payload.get('normalization_stats', {})
        print(f"Checkpoint keys: {list(payload.keys())}")
        print(f"Normalization stats: {stats_dict}")
        # Model summary
        model = MultimodalEmotionModel(num_classes=2)
        model.load_state_dict(payload['model'], strict=False)
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\\nModel Parameters: {total_p:,} total, {train_p:,} trainable")
    else:
        print("No checkpoint found")"""))

# ── 14. Sample Inference ──
cells.append(md("## 🎯 Sample Inference"))
cells.append(code("""\
    if ckpt_path.exists():
        model = model.to(device).eval()
        # Pick a random video
        import random
        test_vid = random.choice(videos)
        print(f"Test video: {test_vid}")
        from emotion_recognition.utils.preprocessing import load_video_tensor, make_sliding_windows
        frames, dur, sfps = load_video_tensor(test_vid, every_n=4, train=False, stage=3)
        windows, indices = make_sliding_windows(frames, window_size=15, stride=8)
        print(f"Frames: {frames.shape}, Windows: {windows.shape}")
        # Zero signal (video-only inference)
        sig = torch.zeros(windows.shape[0], 64, 6)
        with torch.inference_mode():
            windows_d = windows.to(device)
            sig_d = sig.to(device)
            logp, valp, conf = model(windows_d, sig_d)
            probs = torch.exp(logp).mean(dim=0).cpu().numpy()
        class_names = ['negative','positive']
        pred = int(np.argmax(probs))
        print(f"\\nPrediction: {class_names[pred]} (conf={probs[pred]:.4f})")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(class_names, probs, color=['#e74c3c','#2ecc71'], edgecolor='k')
        ax.set_title(f'Inference: {test_vid.name}\\nPredicted: {class_names[pred]}', fontsize=13)
        ax.set_ylim(0, 1)
        plt.tight_layout(); plt.savefig(WORK/'sample_inference.png', dpi=150); plt.show()
        # Show sample frames
        fig, axes = plt.subplots(1, min(5, frames.shape[0]), figsize=(15, 3))
        idxs = np.linspace(0, frames.shape[0]-1, min(5, frames.shape[0]), dtype=int)
        for j, idx in enumerate(idxs):
            fr = frames[idx].permute(1,2,0).numpy()*0.5+0.5
            axes[j].imshow(np.clip(fr, 0, 1)); axes[j].axis('off'); axes[j].set_title(f'Frame {idx}')
        plt.suptitle('Sample Frames', fontsize=14)
        plt.tight_layout(); plt.savefig(WORK/'sample_frames.png', dpi=150); plt.show()
    else:
        print("Skipping inference - no checkpoint")"""))

# ── 15. Save All Outputs ──
cells.append(md("## 💾 Download Outputs"))
cells.append(code("""\
    print("\\n📦 Output files in /kaggle/working/:")
    for f in sorted(WORK.glob('*')):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")
    print("\\n✅ All outputs saved! Download from the Output tab.")"""))

# ── Assemble notebook ──
nb = {
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

out = Path("NeuroBioSense_Kaggle_Pipeline.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Done! Generated {out} ({out.stat().st_size/1024:.0f} KB, {len(cells)} cells)")
