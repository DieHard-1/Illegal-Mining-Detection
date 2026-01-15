"""
Step 3: Compute Spectral Indices (PIXEL-LEVEL)
- For EACH mine polygon → For EACH timestamp → For EACH pixel:
  - Compute NDVI, NBR, NDMI, SI
- This is the core pixel-level computation!
- Output: One .npz file per timestamp with all 4 indices per pixel
"""

import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Create output directory
OUTPUT_DIR = "../data/step3_indices"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("../outputs/validation", exist_ok=True)

print("="*60)
print("STEP 3: COMPUTE SPECTRAL INDICES (PIXEL-LEVEL)")
print("="*60)

# ============================================================================
# SPECTRAL INDEX COMPUTATION FUNCTIONS
# ============================================================================

def compute_ndvi(nir, red):
    """
    NDVI = (NIR - Red) / (NIR + Red)
    Measures vegetation health
    Range: -1 to +1 (higher = more vegetation)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
    return ndvi

def compute_nbr(nir, swir2):
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Normalized Burn Ratio - detects surface disturbance
    Range: -1 to +1 (lower = more disturbance)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        nbr = (nir - swir2) / (nir + swir2)
    return nbr

def compute_ndmi(nir, swir1):
    """
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    Normalized Difference Moisture Index
    Range: -1 to +1 (lower = less moisture)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ndmi = (nir - swir1) / (nir + swir1)
    return ndmi

def compute_si(swir1, swir2):
    """
    SI = (SWIR1 + SWIR2) / 2
    SWIR Index - detects exposed soil/rock
    Range: 0 to 1 (higher = more exposed soil)
    """
    si = (swir1 + swir2) / 2.0
    return si

def compute_all_indices(sentinel_data):
    """
    Compute all 4 spectral indices from Sentinel-2 bands
    
    Input: numpy array of shape (H, W, 5)
           Bands: [B04(Red), B08(NIR), B11(SWIR1), B12(SWIR2), SCL]
           Note: H and W can vary per mine!
    
    Returns: dict with keys 'ndvi', 'nbr', 'ndmi', 'si', 'mask'
             Each value is shape (H, W)
    """
    
    # Extract bands
    B04 = sentinel_data[:, :, 0].astype(np.float32)  # Red
    B08 = sentinel_data[:, :, 1].astype(np.float32)  # NIR
    B11 = sentinel_data[:, :, 2].astype(np.float32)  # SWIR1
    B12 = sentinel_data[:, :, 3].astype(np.float32)  # SWIR2
    SCL = sentinel_data[:, :, 4]                      # Scene Classification
    
    # Create valid pixel mask (not NaN and not cloud)
    valid_mask = ~(np.isnan(B04) | np.isnan(B08) | np.isnan(B11) | np.isnan(B12))
    
    # Compute indices (PIXEL BY PIXEL!)
    ndvi = compute_ndvi(B08, B04)
    nbr = compute_nbr(B08, B12)
    ndmi = compute_ndmi(B08, B11)
    si = compute_si(B11, B12)
    
    # Mask invalid pixels
    ndvi[~valid_mask] = np.nan
    nbr[~valid_mask] = np.nan
    ndmi[~valid_mask] = np.nan
    si[~valid_mask] = np.nan
    
    return {
        'ndvi': ndvi,
        'nbr': nbr,
        'ndmi': ndmi,
        'si': si,
        'mask': valid_mask
    }

# ============================================================================
# PROCESS ALL MINES
# ============================================================================

INPUT_DIR = "../data/step1_sentinel"

# Get list of mines to process
mine_dirs = sorted([d for d in os.listdir(INPUT_DIR) 
                   if os.path.isdir(os.path.join(INPUT_DIR, d))])

print(f"\n[1/3] Found {len(mine_dirs)} mines to process")

# Statistics tracking
all_mine_stats = []

# Process each mine
for mine_id in tqdm(mine_dirs, desc="Computing indices"):
    
    mine_input_path = os.path.join(INPUT_DIR, mine_id)
    mine_output_path = os.path.join(OUTPUT_DIR, mine_id)
    os.makedirs(mine_output_path, exist_ok=True)
    
    # Get all timestamp files
    npy_files = sorted([f for f in os.listdir(mine_input_path) 
                       if f.endswith('.npy')])
    
    if not npy_files:
        print(f"{mine_id}: No data files found, skipping...")
        continue
    
    timestamp_stats = []
    
    # Load metadata to get resolution
    meta_path = os.path.join(mine_input_path, 'metadata.json')
    pixel_resolution_m = 20  # Default
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            pixel_resolution_m = meta.get('actual_resolution_m', 20)
    
    # Process each timestamp
    for npy_file in npy_files:
        timestamp = npy_file.replace('.npy', '')  # e.g., "2023-01-15"
        
        # Load Sentinel-2 data
        input_path = os.path.join(mine_input_path, npy_file)
        
        try:
            sentinel_data = np.load(input_path)
            
            # Check if data is valid
            if sentinel_data.size == 0:
                continue
            
            # PIXEL-LEVEL COMPUTATION!
            # This computes indices for EVERY pixel in the image
            indices = compute_all_indices(sentinel_data)
            
            # Save indices
            output_path = os.path.join(mine_output_path, f"{timestamp}.npz")
            np.savez_compressed(
                output_path,
                ndvi=indices['ndvi'],
                nbr=indices['nbr'],
                ndmi=indices['ndmi'],
                si=indices['si'],
                mask=indices['mask'],
                pixel_size_m=pixel_resolution_m  # Store resolution for later use
            )
            
            # Collect statistics
            valid_pixels = indices['mask'].sum()
            total_pixels = indices['mask'].size
            
            timestamp_stats.append({
                'timestamp': timestamp,
                'image_shape': list(sentinel_data.shape[:2]),  # [H, W]
                'total_pixels': int(total_pixels),
                'valid_pixels': int(valid_pixels),
                'valid_ratio': float(valid_pixels / total_pixels) if total_pixels > 0 else 0,
                'ndvi_mean': float(np.nanmean(indices['ndvi'])) if valid_pixels > 0 else np.nan,
                'ndvi_std': float(np.nanstd(indices['ndvi'])) if valid_pixels > 0 else np.nan,
                'ndvi_min': float(np.nanmin(indices['ndvi'])) if valid_pixels > 0 else np.nan,
                'ndvi_max': float(np.nanmax(indices['ndvi'])) if valid_pixels > 0 else np.nan,
                'nbr_mean': float(np.nanmean(indices['nbr'])) if valid_pixels > 0 else np.nan,
                'ndmi_mean': float(np.nanmean(indices['ndmi'])) if valid_pixels > 0 else np.nan,
                'si_mean': float(np.nanmean(indices['si'])) if valid_pixels > 0 else np.nan
            })
        
        except Exception as e:
            print(f"\nError processing {mine_id}/{timestamp}: {e}")
            continue
    
    if not timestamp_stats:
        continue
    
    # Save per-mine statistics
    with open(os.path.join(mine_output_path, 'indices_stats.json'), 'w') as f:
        json.dump(timestamp_stats, f, indent=2)
    
    # Aggregate stats for this mine
    valid_stats = [s for s in timestamp_stats if not np.isnan(s.get('ndvi_mean', np.nan))]
    
    if valid_stats:
        mine_stats = {
            'mine_id': mine_id,
            'num_timestamps': len(timestamp_stats),
            'avg_valid_ratio': float(np.mean([s['valid_ratio'] for s in valid_stats])),
            'avg_ndvi': float(np.mean([s['ndvi_mean'] for s in valid_stats])),
            'avg_total_pixels': float(np.mean([s['total_pixels'] for s in valid_stats])),
            'pixel_resolution_m': float(pixel_resolution_m),
            'avg_image_height': float(np.mean([s['image_shape'][0] for s in valid_stats])),
            'avg_image_width': float(np.mean([s['image_shape'][1] for s in valid_stats]))
        }
        all_mine_stats.append(mine_stats)

print(f"Processed {len(mine_dirs)} mines")

# ============================================================================
# VALIDATION & STATISTICS
# ============================================================================

print("\n[2/3] Generating validation statistics...")

import pandas as pd
df_stats = pd.DataFrame(all_mine_stats)

print(f"\n{'='*60}")
print("COMPUTATION STATISTICS")
print(f"{'='*60}")
print(f"Total mines processed: {len(df_stats)}")

if not df_stats.empty:
    print(f"\nTimestamps per mine:")
    print(df_stats['num_timestamps'].describe())
    print(f"\nValid pixel ratio (cloud-free):")
    print(df_stats['avg_valid_ratio'].describe())
    print(f"\nNDVI statistics:")
    print(f"  Mean: {df_stats['avg_ndvi'].mean():.3f} ± {df_stats['avg_ndvi'].std():.3f}")
    print(f"  Range: [{df_stats['avg_ndvi'].min():.3f}, {df_stats['avg_ndvi'].max():.3f}]")
    print(f"\nImage dimensions (varies by mine):")
    print(f"  Height: {df_stats['avg_image_height'].mean():.0f} ± {df_stats['avg_image_height'].std():.0f} pixels")
    print(f"  Width: {df_stats['avg_image_width'].mean():.0f} ± {df_stats['avg_image_width'].std():.0f} pixels")
    print(f"\nPixel resolution:")
    print(f"  Min: {df_stats['pixel_resolution_m'].min():.1f} m/pixel")
    print(f"  Max: {df_stats['pixel_resolution_m'].max():.1f} m/pixel")
    print(f"  Mean: {df_stats['pixel_resolution_m'].mean():.1f} m/pixel")
    print(f"\nAverage pixels per mine: {df_stats['avg_total_pixels'].mean():.0f}")

    # Save summary
    df_stats.to_csv("../outputs/validation/step3_indices_stats.csv", index=False)
    print("\nSaved: ../outputs/validation/step3_indices_stats.csv")
else:
    print("No valid data processed!")

# ============================================================================
# SAMPLE VISUALIZATION
# ============================================================================

print("\n[3/3] Creating sample visualizations...")

if not df_stats.empty:
    # Pick a sample mine with good data
    sample_mine_id = df_stats.iloc[0]['mine_id']
    sample_mine_path = os.path.join(OUTPUT_DIR, sample_mine_id)

    # Load first timestamp
    sample_files = sorted([f for f in os.listdir(sample_mine_path) if f.endswith('.npz')])
    if sample_files:
        sample_data = np.load(os.path.join(sample_mine_path, sample_files[0]))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # NDVI
        im0 = axes[0, 0].imshow(sample_data['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0, 0].set_title('NDVI (Vegetation)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
        
        # NBR
        im1 = axes[0, 1].imshow(sample_data['nbr'], cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0, 1].set_title('NBR (Surface Disturbance)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # NDMI
        im2 = axes[0, 2].imshow(sample_data['ndmi'], cmap='Blues', vmin=-1, vmax=1)
        axes[0, 2].set_title('NDMI (Moisture)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # SI
        im3 = axes[1, 0].imshow(sample_data['si'], cmap='YlOrRd', vmin=0, vmax=1)
        axes[1, 0].set_title('SI (Exposed Soil)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Valid mask
        axes[1, 1].imshow(sample_data['mask'], cmap='gray')
        axes[1, 1].set_title('Valid Pixel Mask', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Histograms
        ndvi_valid = sample_data['ndvi'][sample_data['mask']]
        axes[1, 2].hist(ndvi_valid, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('NDVI Distribution', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('NDVI Value')
        axes[1, 2].set_ylabel('Pixel Count')
        axes[1, 2].grid(True, alpha=0.3)
        
        resolution_text = f"Resolution: {sample_data['pixel_size_m']:.1f} m/pixel" if 'pixel_size_m' in sample_data else ""
        plt.suptitle(f'Sample Mine: {sample_mine_id} - {sample_files[0].replace(".npz", "")}\n{resolution_text}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("../outputs/validation/step3_sample_indices.png", dpi=300, bbox_inches='tight')
        print("Saved: ../outputs/validation/step3_sample_indices.png")
        plt.close()

# ============================================================================
# VALIDATION CHECKLIST
# ============================================================================

if not df_stats.empty:
    print("\n" + "="*60)
    print("VALIDATION CHECKLIST")
    print("="*60)

    # Load one sample to verify
    sample_data = np.load(os.path.join(OUTPUT_DIR, df_stats.iloc[0]['mine_id'], 
                                       sorted(os.listdir(os.path.join(OUTPUT_DIR, df_stats.iloc[0]['mine_id'])))[0]))

    checks = [
        ("NDVI range [-1, 1]", 
         np.nanmin(sample_data['ndvi']) >= -1 and np.nanmax(sample_data['ndvi']) <= 1,
         f"[{np.nanmin(sample_data['ndvi']):.2f}, {np.nanmax(sample_data['ndvi']):.2f}]"),
        
        ("NBR range [-1, 1]", 
         np.nanmin(sample_data['nbr']) >= -1 and np.nanmax(sample_data['nbr']) <= 1,
         f"[{np.nanmin(sample_data['nbr']):.2f}, {np.nanmax(sample_data['nbr']):.2f}]"),
        
        ("All 4 indices present",
         all(k in sample_data for k in ['ndvi', 'nbr', 'ndmi', 'si']),
         "NDVI, NBR, NDMI, SI"),
        
        ("Valid mask present",
         'mask' in sample_data,
         f"{sample_data['mask'].sum()} valid pixels"),
        
        ("Pixel-level computation",
         sample_data['ndvi'].ndim == 2,
         f"Shape: {sample_data['ndvi'].shape}"),
        
        ("Resolution stored",
         'pixel_size_m' in sample_data,
         f"{sample_data.get('pixel_size_m', 'N/A')} m/pixel"),
    ]

    for check_name, passed, details in checks:
        status = "True" if passed else "False"
        print(f"{status} {check_name}: {details}")

print("\n" + "="*60)
print("STEP 3 COMPLETE!")
print("="*60)
if not df_stats.empty:
    print(f"\nPixel-level indices computed for {len(df_stats)} mines")
    print(f"Each mine has variable image size (adaptive to mine extent)")
    print(f"Resolution ranges from {df_stats['pixel_resolution_m'].min():.1f} to {df_stats['pixel_resolution_m'].max():.1f} m/pixel")
    print(f"Each timestamp contains 4 indices computed for EVERY pixel:")
    print(f"  - NDVI: Vegetation health (-1 to +1)")
    print(f"  - NBR: Surface disturbance (-1 to +1)")
    print(f"  - NDMI: Moisture content (-1 to +1)")
    print(f"  - SI: Exposed soil (0 to 1)")
    print(f"\nData saved to: {OUTPUT_DIR}/")
else:
    print("\nNo data processed. Check step1 output!")