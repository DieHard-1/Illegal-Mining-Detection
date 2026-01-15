"""
Step 5: Spatial Region Analysis
- Group excavated PIXELS into contiguous REGIONS (mining pits)
- Compute geometric properties: area, perimeter, compactness
- Track how regions grow over time
- Filter noise (very small regions)
"""

import os
import numpy as np
from tqdm import tqdm
import json
from scipy import ndimage
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# Create output directory
OUTPUT_DIR = "../data/step5_regions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("../outputs/validation", exist_ok=True)

print("="*60)
print("STEP 5: SPATIAL REGION ANALYSIS")
print("="*60)

# ============================================================================
# CONNECTED COMPONENT ANALYSIS
# ============================================================================

def identify_regions(excavation_mask, pixel_size_m=20, min_area_m2=400):
    """
    Identify connected regions from excavation mask
    
    Args:
        excavation_mask: Boolean array (H x W) - True = excavated pixel
        pixel_size_m: Size of each pixel in meters
        min_area_m2: Minimum area to keep (filter small noise)
    
    Returns:
        List of region properties
    """
    
    if not excavation_mask.any():
        return []
    
    # Connected component labeling (8-connectivity)
    # This groups adjacent excavated pixels into regions
    labeled_mask, num_regions = label(excavation_mask, connectivity=2, return_num=True)
    
    # Get properties for each region
    regions = regionprops(labeled_mask)
    
    region_stats = []
    
    for region in regions:
        # Compute geometric properties
        area_pixels = region.area
        area_m2 = area_pixels * (pixel_size_m ** 2)
        
        # Skip tiny regions (likely noise)
        if area_m2 < min_area_m2:
            continue
        
        perimeter_pixels = region.perimeter
        perimeter_m = perimeter_pixels * pixel_size_m
        
        # Compactness: 1 = perfect circle, lower = irregular
        if perimeter_m > 0:
            compactness = (4 * np.pi * area_m2) / (perimeter_m ** 2)
        else:
            compactness = 0
        
        # Store properties
        region_stats.append({
            'label': int(region.label),
            'area_m2': float(area_m2),
            'area_pixels': int(area_pixels),
            'perimeter_m': float(perimeter_m),
            'compactness': float(compactness),
            'centroid_row': float(region.centroid[0]),
            'centroid_col': float(region.centroid[1]),
            'bbox': [int(x) for x in region.bbox],  # (min_row, min_col, max_row, max_col)
            'eccentricity': float(region.eccentricity),
            'solidity': float(region.solidity)
        })
    
    return region_stats

# ============================================================================
# TEMPORAL GROWTH TRACKING
# ============================================================================

def track_temporal_growth(change_dir, indices_dir, mine_id):
    """
    Track how excavation regions grow over time
    
    Returns:
        Dictionary with temporal region info
    """
    
    mine_path = os.path.join(change_dir, mine_id)
    
    # Get pixel resolution from indices metadata
    indices_path = os.path.join(indices_dir, mine_id)
    pixel_size_m = 20  # Default
    
    if os.path.exists(indices_path):
        npz_files = [f for f in os.listdir(indices_path) if f.endswith('.npz')]
        if npz_files:
            sample_data = np.load(os.path.join(indices_path, npz_files[0]))
            if 'pixel_size_m' in sample_data:
                pixel_size_m = float(sample_data['pixel_size_m'])
    
    # Get all change files (sorted chronologically)
    change_files = sorted([f for f in os.listdir(mine_path) 
                          if f.endswith('.npz') and f.startswith('change')])
    
    temporal_regions = []
    
    for change_file in change_files:
        # Load change detection
        data = np.load(os.path.join(mine_path, change_file))
        
        excavation_mask = data['excavation_mask']
        ts1 = str(data['ts1'])
        ts2 = str(data['ts2'])
        
        # Identify regions with actual pixel size
        regions = identify_regions(excavation_mask, pixel_size_m=pixel_size_m)
        
        # Aggregate stats
        total_area = sum(r['area_m2'] for r in regions)
        num_regions = len(regions)
        
        temporal_regions.append({
            'change_file': change_file,
            'ts1': ts1,
            'ts2': ts2,
            'num_regions': num_regions,
            'total_area_m2': total_area,
            'regions': regions
        })
    
    # Compute growth metrics
    areas = [tr['total_area_m2'] for tr in temporal_regions]
    
    if len(areas) > 1:
        # Growth between consecutive time periods
        growth = [areas[i] - areas[i-1] for i in range(1, len(areas))]
        
        # Growth consistency: % of periods with non-negative growth
        non_negative = sum(1 for g in growth if g >= 0)
        growth_consistency = non_negative / len(growth) if growth else 0
        
        total_growth = areas[-1] - areas[0]
    else:
        growth_consistency = 0
        total_growth = 0
    
    return {
        'mine_id': mine_id,
        'temporal_regions': temporal_regions,
        'growth_consistency': growth_consistency,
        'total_growth_m2': total_growth,
        'max_area_m2': max(areas) if areas else 0,
        'num_time_periods': len(temporal_regions)
    }

# ============================================================================
# PROCESS ALL MINES
# ============================================================================

INPUT_DIR = "../data/step4_changes"
INDICES_DIR = "../data/step3_indices"  # To get pixel resolution

mine_dirs = sorted([d for d in os.listdir(INPUT_DIR) 
                   if os.path.isdir(os.path.join(INPUT_DIR, d))])

print(f"\n[1/3] Found {len(mine_dirs)} mines to process")

all_results = {}

# Process each mine
for mine_id in tqdm(mine_dirs, desc="Spatial analysis"):
    
    # Track temporal growth with actual resolution
    result = track_temporal_growth(INPUT_DIR, INDICES_DIR, mine_id)
    all_results[mine_id] = result
    
    # Save per-mine results
    output_file = os.path.join(OUTPUT_DIR, f"{mine_id}_regions.json")
    
    # Convert to JSON-serializable format
    result_json = {
        'mine_id': result['mine_id'],
        'growth_consistency': result['growth_consistency'],
        'total_growth_m2': result['total_growth_m2'],
        'max_area_m2': result['max_area_m2'],
        'num_time_periods': result['num_time_periods'],
        'temporal_regions': result['temporal_regions']
    }
    
    with open(output_file, 'w') as f:
        json.dump(result_json, f, indent=2)

print(f"Processed {len(mine_dirs)} mines")

# ============================================================================
# VALIDATION & STATISTICS
# ============================================================================

print("\n[2/3] Generating validation statistics...")

import pandas as pd

stats_list = []
for mine_id, result in all_results.items():
    stats_list.append({
        'mine_id': mine_id,
        'num_time_periods': result['num_time_periods'],
        'total_growth_m2': result['total_growth_m2'],
        'max_area_m2': result['max_area_m2'],
        'growth_consistency': result['growth_consistency'],
        'avg_regions_per_period': np.mean([tr['num_regions'] for tr in result['temporal_regions']])
    })

df_stats = pd.DataFrame(stats_list)

print(f"\n{'='*60}")
print("SPATIAL ANALYSIS STATISTICS")
print(f"{'='*60}")
print(f"Total mines: {len(df_stats)}")
print(f"\nTime periods per mine:")
print(df_stats['num_time_periods'].describe())
print(f"\nTotal growth (m²):")
print(df_stats['total_growth_m2'].describe())
print(f"\nGrowth consistency (0-1):")
print(df_stats['growth_consistency'].describe())
print(f"\nMines with >50% consistent growth: {(df_stats['growth_consistency'] > 0.5).sum()}")
print(f"\nAverage regions per time period:")
print(df_stats['avg_regions_per_period'].describe())

# Save summary
df_stats.to_csv("../outputs/validation/step5_spatial_stats.csv", index=False)
print("\nSaved: ../outputs/validation/step5_spatial_stats.csv")

# ============================================================================
# SAMPLE VISUALIZATION
# ============================================================================

print("\n[3/3] Creating sample visualizations...")

# Pick mine with significant and consistent growth
df_filtered = df_stats[(df_stats['growth_consistency'] > 0.5) & (df_stats['total_growth_m2'] > 0)]

if len(df_filtered) > 0:
    sample_mine_id = df_filtered.nlargest(1, 'total_growth_m2').iloc[0]['mine_id']
    sample_result = all_results[sample_mine_id]
    
    # Plot temporal growth
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total area over time
    time_points = range(len(sample_result['temporal_regions']))
    areas = [tr['total_area_m2'] for tr in sample_result['temporal_regions']]
    
    axes[0, 0].plot(time_points, areas, marker='o', linewidth=2, markersize=8, color='red')
    axes[0, 0].fill_between(time_points, 0, areas, alpha=0.3, color='red')
    axes[0, 0].set_xlabel('Time Period', fontsize=12)
    axes[0, 0].set_ylabel('Excavated Area (m²)', fontsize=12)
    axes[0, 0].set_title(f'Temporal Growth Profile\nTotal Growth: {sample_result["total_growth_m2"]:.0f} m²', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Number of regions over time
    num_regions = [tr['num_regions'] for tr in sample_result['temporal_regions']]
    
    axes[0, 1].plot(time_points, num_regions, marker='s', linewidth=2, markersize=8, color='blue')
    axes[0, 1].set_xlabel('Time Period', fontsize=12)
    axes[0, 1].set_ylabel('Number of Regions', fontsize=12)
    axes[0, 1].set_title(f'Number of Excavation Regions\nGrowth Consistency: {sample_result["growth_consistency"]:.1%}', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Growth rate (derivative)
    if len(areas) > 1:
        growth_rates = [areas[i] - areas[i-1] for i in range(1, len(areas))]
        axes[1, 0].bar(range(1, len(areas)), growth_rates, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_xlabel('Time Period', fontsize=12)
        axes[1, 0].set_ylabel('Growth Rate (m²/period)', fontsize=12)
        axes[1, 0].set_title('Incremental Area Change', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Region size distribution (latest timestamp)
    latest_regions = sample_result['temporal_regions'][-1]['regions']
    if latest_regions:
        region_areas = [r['area_m2'] for r in latest_regions]
        axes[1, 1].hist(region_areas, bins=20, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Region Area (m²)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title(f'Region Size Distribution (Latest)\n{len(latest_regions)} regions', 
                           fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Mine: {sample_mine_id} - Temporal Spatial Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("../outputs/validation/step5_temporal_growth.png", dpi=300, bbox_inches='tight')
    print("Saved: ../outputs/validation/step5_temporal_growth.png")
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
    
    ("Regions identified",
     df_stats['avg_regions_per_period'].sum() > 0,
     f"{df_stats['avg_regions_per_period'].mean():.1f} avg regions/period"),
    
    ("Growth tracked",
     (df_stats['total_growth_m2'] != 0).sum() > 0,
     f"{(df_stats['total_growth_m2'] > 0).sum()} mines with growth"),
    
    ("Consistent growth detected",
     (df_stats['growth_consistency'] > 0.5).sum() > 0,
     f"{(df_stats['growth_consistency'] > 0.5).sum()} mines with consistent growth"),
    
    ("Region files saved",
     len(os.listdir(OUTPUT_DIR)) > 0,
     f"{len(os.listdir(OUTPUT_DIR))} region files"),
]

for check_name, passed, details in checks:
    status = "True" if passed else "False"
    print(f"{status} {check_name}: {details}")

print("\n" + "="*60)
print("STEP 5 COMPLETE!")
print("="*60)
print(f"\nSpatial analysis completed!")
print(f"Excavated pixels grouped into contiguous regions")
print(f"Temporal growth tracked for each mine")
print(f"\nKey findings:")
print(f"  - Mines with activity: {(df_stats['total_growth_m2'] > 0).sum()}")
print(f"  - Mines with consistent growth: {(df_stats['growth_consistency'] > 0.5).sum()}")
print(f"  - Average max area: {df_stats['max_area_m2'].mean():.0f} m²")
print(f"\nData saved to: {OUTPUT_DIR}/")