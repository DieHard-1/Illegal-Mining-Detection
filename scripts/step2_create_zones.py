"""
Step 2: Create Legal and No-Go Zone Assignment
- Assign 30% of ENTIRE mines as NO-GO zones (prohibited)
- Assign 70% of ENTIRE mines as LEGAL zones (allowed)
- Strategy: Mixed approach (small + large mines in no-go)
"""

import os
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create output directory
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../outputs/validation", exist_ok=True)

print("="*60)
print("STEP 2: CREATE LEGAL AND NO-GO ZONE ASSIGNMENT")
print("="*60)

# ============================================================================
# LOAD VALIDATED MINES
# ============================================================================

print("\n[1/5] Loading validated mine polygons...")
mines = gpd.read_file("../data/processed/mines_validated.shp")

print(f"Loaded {len(mines)} mines")
print(f"CRS: {mines.crs}")

# --- FIX: ENSURE CENTROIDS AND AREA EXIST ---
if 'area_km2' not in mines.columns:
    mines['area_km2'] = mines.geometry.area * (111 ** 2)

# Recalculate centroids to ensure they exist in the GeoDataFrame
mines['centroid_lon'] = mines.geometry.centroid.x
mines['centroid_lat'] = mines.geometry.centroid.y
# --------------------------------------------

# ============================================================================
# ZONE ASSIGNMENT STRATEGY
# ============================================================================

print("\n[2/5] Assigning zones (30% NO-GO, 70% LEGAL)...")

# Strategy: Mixed approach
# - Select 15% smallest mines → NO-GO (simulate protected small areas)
# - Select 15% largest mines → NO-GO (simulate large restricted zones)
# - Remaining 70% → LEGAL

total_mines = len(mines)
nogo_count = int(0.30 * total_mines)  # 30% as no-go
legal_count = total_mines - nogo_count  # 70% as legal

print(f"\nTarget distribution:")
print(f"  NO-GO mines: {nogo_count} (30%)")
print(f"  LEGAL mines: {legal_count} (70%)")

# Sort by area
mines_sorted = mines.sort_values('area_km2').reset_index(drop=True)

# Assign zones
mines_sorted['zone_type'] = 'legal'  # Default all to legal

# Bottom 15% → NO-GO (small mines)
bottom_15pct = int(0.15 * total_mines)
mines_sorted.loc[0:bottom_15pct-1, 'zone_type'] = 'nogo'

# Top 15% → NO-GO (large mines)
top_15pct = int(0.15 * total_mines)
mines_sorted.loc[total_mines-top_15pct:, 'zone_type'] = 'nogo'

# Verify counts
actual_nogo = (mines_sorted['zone_type'] == 'nogo').sum()
actual_legal = (mines_sorted['zone_type'] == 'legal').sum()

print(f"\nActual distribution:")
print(f"  NO-GO mines: {actual_nogo} ({actual_nogo/total_mines*100:.1f}%)")
print(f"  LEGAL mines: {actual_legal} ({actual_legal/total_mines*100:.1f}%)")

# ============================================================================
# CREATE SEPARATE GEODATAFRAMES
# ============================================================================

print("\n[3/5] Creating zone GeoDataFrames...")

# NO-GO zones (entire mine polygons that are prohibited)
nogo_zones = mines_sorted[mines_sorted['zone_type'] == 'nogo'].copy()
nogo_zones['zone_category'] = 'no-go'
nogo_zones['description'] = 'Mining prohibited in this entire area'

# LEGAL zones (entire mine polygons that are allowed)
legal_zones = mines_sorted[mines_sorted['zone_type'] == 'legal'].copy()
legal_zones['zone_category'] = 'legal'
legal_zones['description'] = 'Mining allowed in this entire area'

print(f"NO-GO zones: {len(nogo_zones)} mines")
print(f"LEGAL zones: {len(legal_zones)} mines")

# Statistics
nogo_area_total = nogo_zones['area_km2'].sum()
legal_area_total = legal_zones['area_km2'].sum()
total_area = mines['area_km2'].sum()

print(f"\nArea distribution:")
print(f"  Total area: {total_area:.2f} km²")
print(f"  NO-GO area: {nogo_area_total:.2f} km² ({nogo_area_total/total_area*100:.1f}%)")
print(f"  LEGAL area: {legal_area_total:.2f} km² ({legal_area_total/total_area*100:.1f}%)")

print(f"\nNO-GO zone characteristics:")
print(f"  Area range: {nogo_zones['area_km2'].min():.4f} - {nogo_zones['area_km2'].max():.4f} km²")
print(f"  Mean area: {nogo_zones['area_km2'].mean():.4f} km²")

print(f"\nLEGAL zone characteristics:")
print(f"  Area range: {legal_zones['area_km2'].min():.4f} - {legal_zones['area_km2'].max():.4f} km²")
print(f"  Mean area: {legal_zones['area_km2'].mean():.4f} km²")

# ============================================================================
# SAVE SHAPEFILES
# ============================================================================

print("\n[4/5] Saving shapefiles...")

# Save NO-GO zones
nogo_zones.to_file("../data/processed/nogo_zones.shp")
print("Saved: ../data/processed/nogo_zones.shp")

# Save LEGAL zones
legal_zones.to_file("../data/processed/legal_zones.shp")
print("Saved: ../data/processed/legal_zones.shp")

# Save combined with zone labels
mines_sorted.to_file("../data/processed/mines_with_zones.shp")
print("Saved: ../data/processed/mines_with_zones.shp")

# Save CSV for easy reference
zone_summary = pd.DataFrame({
    'mine_id': mines_sorted['mine_id'],
    'zone_type': mines_sorted['zone_type'],
    'area_km2': mines_sorted['area_km2'],
    'centroid_lon': mines_sorted['centroid_lon'],
    'centroid_lat': mines_sorted['centroid_lat']
})
zone_summary.to_csv("../outputs/validation/step2_zone_assignments.csv", index=False)
print("Saved: ../outputs/validation/step2_zone_assignments.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n[5/5] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: All zones with color
legal_zones.plot(ax=axes[0, 0], color='green', alpha=0.5, edgecolor='darkgreen', linewidth=0.5, label='LEGAL')
nogo_zones.plot(ax=axes[0, 0], color='red', alpha=0.5, edgecolor='darkred', linewidth=0.5, label='NO-GO')
axes[0, 0].set_title(f'Zone Assignment: {actual_legal} LEGAL (Green) vs {actual_nogo} NO-GO (Red)', 
                    fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Longitude', fontsize=12)
axes[0, 0].set_ylabel('Latitude', fontsize=12)
axes[0, 0].legend(fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Area distribution by zone
axes[0, 1].hist([legal_zones['area_km2'], nogo_zones['area_km2']], 
               bins=30, 
               label=['LEGAL', 'NO-GO'],
               color=['green', 'red'],
               alpha=0.6,
               edgecolor='black')
axes[0, 1].set_xlabel('Area (km²)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Area Distribution by Zone Type', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Sample detail (zoom into one region)
# Get spatial bounds
all_bounds = mines.total_bounds
mid_lon = (all_bounds[0] + all_bounds[2]) / 2
mid_lat = (all_bounds[1] + all_bounds[3]) / 2

# Find mines near center
mines_sorted['dist_to_center'] = np.sqrt(
    (mines_sorted['centroid_lon'] - mid_lon)**2 + 
    (mines_sorted['centroid_lat'] - mid_lat)**2
)
sample_mines = mines_sorted.nsmallest(30, 'dist_to_center')

sample_legal = sample_mines[sample_mines['zone_type'] == 'legal']
sample_nogo = sample_mines[sample_mines['zone_type'] == 'nogo']

sample_legal.plot(ax=axes[1, 0], color='green', alpha=0.6, edgecolor='darkgreen', linewidth=1.5)
sample_nogo.plot(ax=axes[1, 0], color='red', alpha=0.6, edgecolor='darkred', linewidth=1.5)
axes[1, 0].set_title('Sample Region Detail (30 mines)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Longitude', fontsize=12)
axes[1, 0].set_ylabel('Latitude', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Statistics summary
axes[1, 1].axis('off')
summary_text = f"""
ZONE ASSIGNMENT SUMMARY

Total Mines: {total_mines}

NO-GO ZONES (30%):
  • Count: {actual_nogo} mines
  • Total Area: {nogo_area_total:.2f} km²
  • Area Range: {nogo_zones['area_km2'].min():.4f} - {nogo_zones['area_km2'].max():.4f} km²
  • Composition:
    - {bottom_15pct} smallest mines (bottom 15%)
    - {top_15pct} largest mines (top 15%)

LEGAL ZONES (70%):
  • Count: {actual_legal} mines
  • Total Area: {legal_area_total:.2f} km²
  • Area Range: {legal_zones['area_km2'].min():.4f} - {legal_zones['area_km2'].max():.4f} km²
  • Composition:
    - Middle 70% of mines by area

STRATEGY:
Mixed approach tests detection on:
  Small protected areas (small no-go mines)
  Large restricted zones (large no-go mines)
  Active mining areas (legal zones)

NEXT STEP:
In Step 6, any excavation detected in 
NO-GO mines will be flagged as ILLEGAL
"""

axes[1, 1].text(0.1, 0.9, summary_text, 
                transform=axes[1, 1].transAxes,
                fontsize=11,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("../outputs/validation/step2_zone_assignment.png", dpi=300, bbox_inches='tight')
print("Saved: ../outputs/validation/step2_zone_assignment.png")
plt.close()

# Create individual mine examples
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, ax in enumerate(axes):
    if i < len(nogo_zones):
        sample_nogo = nogo_zones.iloc[i:i+1]
        sample_nogo.plot(ax=ax, color='red', alpha=0.6, edgecolor='darkred', linewidth=2)
        ax.set_title(f'NO-GO Mine Example {i+1}\n{sample_nogo.iloc[0]["mine_id"]}\nArea: {sample_nogo.iloc[0]["area_km2"]:.4f} km²',
                    fontsize=12, fontweight='bold', color='darkred')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)

plt.suptitle('NO-GO Zone Examples (Entire Mines are Prohibited)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("../outputs/validation/step2_nogo_examples.png", dpi=300, bbox_inches='tight')
print("Saved: ../outputs/validation/step2_nogo_examples.png")
plt.close()

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

print("\n" + "="*60)
print("VALIDATION CHECKS")
print("="*60)

# Check 1: Count
count_check = (actual_nogo + actual_legal) == total_mines
print(f"✓ Total count check: {actual_nogo} + {actual_legal} = {total_mines} {'PASS' if count_check else 'FAIL'}")

# Check 2: No overlaps (each mine in only one zone)
mine_ids_nogo = set(nogo_zones['mine_id'])
mine_ids_legal = set(legal_zones['mine_id'])
overlap = mine_ids_nogo & mine_ids_legal
print(f"✓ No overlaps: {len(overlap) == 0} {'PASS - no mine in both zones' if len(overlap) == 0 else 'FAIL'}")

# Check 3: All mines accounted for
all_mine_ids = set(mines['mine_id'])
assigned_ids = mine_ids_nogo | mine_ids_legal
missing = all_mine_ids - assigned_ids
print(f"✓ All mines assigned: {len(missing) == 0} {'PASS - all mines have zones' if len(missing) == 0 else f'FAIL - {len(missing)} missing'}")

# Check 4: Zone type values
zone_types_correct = set(mines_sorted['zone_type'].unique()) == {'legal', 'nogo'}
print(f"✓ Zone types correct: {zone_types_correct} {'PASS' if zone_types_correct else 'FAIL'}")

print("\n" + "="*60)
print("STEP 2 COMPLETE!")
print("="*60)
print(f"\nCreated zone assignments:")
print(f"  NO-GO zones: ../data/processed/nogo_zones.shp ({actual_nogo} ENTIRE mines)")
print(f"  LEGAL zones: ../data/processed/legal_zones.shp ({actual_legal} ENTIRE mines)")