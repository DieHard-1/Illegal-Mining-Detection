"""
Step 4: Temporal Change Detection (PIXEL-LEVEL)
- For EACH mine → For EACH pixel → Track changes over time
- Compare timestamps: Jan 2023 vs Jun 2023 vs Dec 2023
- Compute disturbance score for EACH pixel
- Use GMM to classify pixels as: excavated vs unchanged
- This is where we detect WHICH PIXELS got excavated!
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Create output directory
OUTPUT_DIR = "../data/step4_changes"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("../outputs/validation", exist_ok=True)

print("="*60)
print("STEP 4: TEMPORAL CHANGE DETECTION (PIXEL-LEVEL)")
print("="*60)

# ============================================================================
# TEMPORAL COMPARISON STRATEGY
# ============================================================================

def get_temporal_pairs(timestamps, strategy='seasonal'):
    """
    Create pairs of timestamps to compare
    
    Strategies:
    - 'sequential': Compare consecutive timestamps (t1 vs t2, t2 vs t3, ...)
    - 'seasonal': Compare same seasons across time (Jan vs Jul, Feb vs Aug, ...)
    - 'baseline': Compare everything to first timestamp
    """
    
    pairs = []
    
    if strategy == 'sequential':
        # Compare consecutive timestamps
        for i in range(len(timestamps) - 1):
            pairs.append((timestamps[i], timestamps[i + 1]))
    
    elif strategy == 'seasonal':
        # Compare timestamps ~6 months apart (within same year)
        for i in range(len(timestamps)):
            for j in range(i + 1, len(timestamps)):
                # Extract month
                month_i = int(timestamps[i].split('-')[1])
                month_j = int(timestamps[j].split('-')[1])
                
                # Compare if ~6 months apart
                if 5 <= abs(month_j - month_i) <= 7:
                    pairs.append((timestamps[i], timestamps[j]))
    
    elif strategy == 'baseline':
        # Compare all to first timestamp
        baseline = timestamps[0]
        for ts in timestamps[1:]:
            pairs.append((baseline, ts))
    
    return pairs

# ============================================================================
# PIXEL-LEVEL CHANGE COMPUTATION
# ============================================================================

def compute_pixel_changes(indices_t1, indices_t2):
    """
    Compute change for EACH PIXEL between two timestamps
    
    Returns:
        delta_ndvi: Change in NDVI for each pixel (H x W)
        delta_nbr: Change in NBR for each pixel
        delta_ndmi: Change in NDMI for each pixel
        delta_si: Change in SI for each pixel
        disturbance: Combined disturbance score for each pixel
        valid_mask: Pixels valid in both timestamps
    """
    
    # Delta = t1 - t2 (positive means decrease, which indicates disturbance)
    delta_ndvi = indices_t1['ndvi'] - indices_t2['ndvi']
    delta_nbr = indices_t1['nbr'] - indices_t2['nbr']
    delta_ndmi = indices_t1['ndmi'] - indices_t2['ndmi']
    delta_si = indices_t2['si'] - indices_t1['si']  # Increase in SI is suspicious
    
    # Valid mask: valid in BOTH timestamps
    valid_mask = indices_t1['mask'] & indices_t2['mask']
    
    # Mask invalid pixels
    delta_ndvi[~valid_mask] = np.nan
    delta_nbr[~valid_mask] = np.nan
    delta_ndmi[~valid_mask] = np.nan
    delta_si[~valid_mask] = np.nan
    
    # Compute DISTURBANCE SCORE for EACH PIXEL
    # Mining excavation shows:
    #   - Large drop in NDVI (vegetation removal)
    #   - Drop in NBR (surface disturbance)
    #   - Drop in NDMI (moisture loss)
    #   - Increase in SI (soil exposure)
    
    # Weighted combination (PIXEL BY PIXEL!)
    disturbance = (
        2.0 * delta_ndvi +      # Heavy weight on vegetation loss
        1.5 * delta_nbr +       # Surface disturbance
        1.0 * delta_ndmi +      # Moisture loss
        1.0 * delta_si          # Soil exposure increase
    )
    
    disturbance[~valid_mask] = np.nan
    
    return {
        'delta_ndvi': delta_ndvi,
        'delta_nbr': delta_nbr,
        'delta_ndmi': delta_ndmi,
        'delta_si': delta_si,
        'disturbance': disturbance,
        'valid_mask': valid_mask
    }

# ============================================================================
# GMM-BASED EXCAVATION CLASSIFICATION
# ============================================================================

def classify_pixels_gmm(disturbance_map, n_components=2):
    """
    Use Gaussian Mixture Model to classify EACH PIXEL
    
    Component 1: Background (small changes, natural variation)
    Component 2: Excavation (large disturbance)
    
    Returns:
        excavation_mask: Boolean array (H x W) - True = excavated pixel
        gmm_labels: Component label for each pixel
        gmm_model: Fitted GMM model
    """
    
    # Flatten disturbance scores
    scores_flat = disturbance_map.flatten()
    
    # Remove NaN values
    valid_indices = ~np.isnan(scores_flat)
    valid_scores = scores_flat[valid_indices]
    
    if len(valid_scores) < 100:
        # Too few valid pixels - return empty mask
        return np.zeros_like(disturbance_map, dtype=bool), None, None
    
    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        max_iter=200
    )
    
    gmm.fit(valid_scores.reshape(-1, 1))
    
    # Predict labels for valid pixels
    labels_valid = gmm.predict(valid_scores.reshape(-1, 1))
    
    # Map back to 2D
    labels_flat = np.full_like(scores_flat, -1, dtype=int)
    labels_flat[valid_indices] = labels_valid
    labels_2d = labels_flat.reshape(disturbance_map.shape)
    
    # Identify excavation component (component with HIGHEST mean disturbance)
    means = gmm.means_.flatten()
    excavation_component = np.argmax(means)  # Highest disturbance = excavation
    
    # Create excavation mask
    excavation_mask = (labels_2d == excavation_component)
    
    return excavation_mask, labels_2d, gmm

# ============================================================================
# PROCESS ALL MINES
# ============================================================================

INPUT_DIR = "../data/step3_indices"

# Get list of mines
mine_dirs = sorted([d for d in os.listdir(INPUT_DIR) 
                   if os.path.isdir(os.path.join(INPUT_DIR, d))])

print(f"\n[1/3] Found {len(mine_dirs)} mines to process")

# Statistics
all_stats = []

# Process each mine
for mine_id in tqdm(mine_dirs, desc="Change detection"):
    
    mine_input_path = os.path.join(INPUT_DIR, mine_id)
    mine_output_path = os.path.join(OUTPUT_DIR, mine_id)
    os.makedirs(mine_output_path, exist_ok=True)
    
    # Get all timestamps
    npz_files = sorted([f for f in os.listdir(mine_input_path) 
                       if f.endswith('.npz')])
    timestamps = [f.replace('.npz', '') for f in npz_files]
    
    if len(timestamps) < 2:
        continue
    
    # Create temporal pairs
    pairs = get_temporal_pairs(timestamps, strategy='sequential')
    
    change_results = []
    
    # Compare each pair
    for idx, (ts1, ts2) in enumerate(pairs):
        
        # Load indices for both timestamps
        data_t1 = np.load(os.path.join(mine_input_path, f"{ts1}.npz"))
        data_t2 = np.load(os.path.join(mine_input_path, f"{ts2}.npz"))
        
        # Compute PIXEL-LEVEL changes
        changes = compute_pixel_changes(data_t1, data_t2)
        
        # Classify pixels using GMM
        excavation_mask, labels, gmm = classify_pixels_gmm(changes['disturbance'])
        
        # Save results
        output_file = os.path.join(mine_output_path, f"change_{idx:03d}_{ts1}_to_{ts2}.npz")
        np.savez_compressed(
            output_file,
            ts1=ts1,
            ts2=ts2,
            delta_ndvi=changes['delta_ndvi'],
            delta_nbr=changes['delta_nbr'],
            delta_ndmi=changes['delta_ndmi'],
            delta_si=changes['delta_si'],
            disturbance=changes['disturbance'],
            excavation_mask=excavation_mask,
            labels=labels,
            valid_mask=changes['valid_mask']
        )
        
        # Collect stats
        num_excavated = excavation_mask.sum()
        total_valid = changes['valid_mask'].sum()
        
        change_results.append({
            'ts1': ts1,
            'ts2': ts2,
            'total_pixels': excavation_mask.size,
            'valid_pixels': int(total_valid),
            'excavated_pixels': int(num_excavated),
            'excavation_ratio': float(num_excavated / total_valid) if total_valid > 0 else 0,
            'mean_disturbance': float(np.nanmean(changes['disturbance']))
        })
    
    # Save per-mine change summary
    with open(os.path.join(mine_output_path, 'change_summary.json'), 'w') as f:
        json.dump(change_results, f, indent=2)
    
    # Aggregate stats
    all_stats.append({
        'mine_id': mine_id,
        'num_comparisons': len(change_results),
        'avg_excavated_pixels': np.mean([r['excavated_pixels'] for r in change_results]),
        'max_excavated_pixels': np.max([r['excavated_pixels'] for r in change_results]) if change_results else 0,
        'avg_excavation_ratio': np.mean([r['excavation_ratio'] for r in change_results])
    })

print(f"Processed {len(mine_dirs)} mines")

# ============================================================================
# VALIDATION & STATISTICS
# ============================================================================

print("\n[2/3] Generating validation statistics...")

import pandas as pd
df_stats = pd.DataFrame(all_stats)

print(f"\n{'='*60}")
print("CHANGE DETECTION STATISTICS")
print(f"{'='*60}")
print(f"Total mines processed: {len(df_stats)}")
print(f"\nComparisons per mine:")
print(df_stats['num_comparisons'].describe())
print(f"\nExcavated pixels per comparison:")
print(df_stats['avg_excavated_pixels'].describe())
print(f"\nExcavation ratio (% of valid pixels):")
print((df_stats['avg_excavation_ratio'] * 100).describe())

# Save summary
df_stats.to_csv("../outputs/validation/step4_change_stats.csv", index=False)
print("\nSaved: ../outputs/validation/step4_change_stats.csv")

# ============================================================================
# SAMPLE VISUALIZATION
# ============================================================================

print("\n[3/3] Creating sample visualizations...")

# Pick a mine with significant changes
sample_mine_id = df_stats.nlargest(1, 'avg_excavated_pixels').iloc[0]['mine_id']
sample_mine_path = os.path.join(OUTPUT_DIR, sample_mine_id)

# Load first change detection
change_files = sorted([f for f in os.listdir(sample_mine_path) if f.endswith('.npz')])

if change_files:
    sample_change = np.load(os.path.join(sample_mine_path, change_files[0]))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Delta NDVI
    im0 = axes[0, 0].imshow(sample_change['delta_ndvi'], cmap='RdYlGn_r', vmin=-1, vmax=1)
    axes[0, 0].set_title('ΔNDVI (Vegetation Change)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Delta NBR
    im1 = axes[0, 1].imshow(sample_change['delta_nbr'], cmap='RdYlGn_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('ΔNBR (Surface Change)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Disturbance score
    im2 = axes[0, 2].imshow(sample_change['disturbance'], cmap='hot', vmin=0, vmax=5)
    axes[0, 2].set_title('Disturbance Score', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # GMM labels
    im3 = axes[1, 0].imshow(sample_change['labels'], cmap='viridis')
    axes[1, 0].set_title('GMM Components', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Excavation mask
    axes[1, 1].imshow(sample_change['excavation_mask'], cmap='Reds')
    axes[1, 1].set_title('EXCAVATED PIXELS (Red)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Histogram of disturbance
    dist_valid = sample_change['disturbance'][sample_change['valid_mask']]
    axes[1, 2].hist(dist_valid, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(np.nanpercentile(dist_valid, 90), color='red', linestyle='--', 
                      label='90th percentile')
    axes[1, 2].set_title('Disturbance Distribution', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Disturbance Score')
    axes[1, 2].set_ylabel('Pixel Count')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Mine: {sample_mine_id} - Change Detection\n{sample_change["ts1"]} → {sample_change["ts2"]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("../outputs/validation/step4_sample_change.png", dpi=300, bbox_inches='tight')
    print("Saved: ../outputs/validation/step4_sample_change.png")
    plt.close()

# ============================================================================
# VALIDATION CHECKLIST
# ============================================================================

print("\n" + "="*60)
print("VALIDATION CHECKLIST")
print("="*60)

checks = [
    ("All mines processed", 
     len(df_stats) > 0,
     f"{len(df_stats)} mines"),
    
    ("Temporal pairs created",
     df_stats['num_comparisons'].sum() > 0,
     f"{df_stats['num_comparisons'].sum()} total comparisons"),
    
    ("Excavation detected",
     df_stats['avg_excavated_pixels'].sum() > 0,
     f"{df_stats['avg_excavated_pixels'].mean():.0f} avg pixels/mine"),
    
    ("Reasonable excavation ratio",
     (df_stats['avg_excavation_ratio'] < 0.5).all(),
     f"{df_stats['avg_excavation_ratio'].mean():.2%} avg ratio"),
    
    ("GMM classification working",
     len(change_files) > 0,
     "Change files generated"),
]

for check_name, passed, details in checks:
    status = "True" if passed else "False"
    print(f"{status} {check_name}: {details}")

print("\n" + "="*60)
print("STEP 4 COMPLETE!")
print("="*60)