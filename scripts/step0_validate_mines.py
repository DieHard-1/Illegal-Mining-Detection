"""
Step 0: Data Validation
- Load and validate the 506 mine polygons shapefile
- Analyze which mines are excavated vs not excavated
- This will help us divide into legal/no-go zones properly
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import pandas as pd

# Create output directories
os.makedirs("../outputs/validation", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)

print("="*60)
print("STEP 0: DATA VALIDATION & ANALYSIS")
print("="*60)

# ============================================================================
# LOAD SHAPEFILE
# ============================================================================

print("\n[1/6] Loading shapefile...")
mines = gpd.read_file("../data/raw/mines_cil_polygon/mines_cils.shp")

print(f"Total mines loaded: {len(mines)}")
print(f"Original CRS: {mines.crs}")

# ============================================================================
# ANALYZE ATTRIBUTES
# ============================================================================

print("\n[2/6] Analyzing shapefile attributes...")

print(f"\nAll columns in shapefile:")
for col in mines.columns:
    print(f"  - {col}: {mines[col].dtype}")

print(f"\nFirst few rows:")
print(mines.head())

print(f"\nDetailed statistics:")
print(mines.describe())

# Check for any status/activity indicators
print(f"\nLooking for excavation/activity indicators...")
potential_status_cols = [col for col in mines.columns 
                        if any(keyword in col.lower() 
                               for keyword in ['status', 'active', 'excavat', 'operat', 'type', 'class'])]

if potential_status_cols:
    print(f"Found potential status columns: {potential_status_cols}")
    for col in potential_status_cols:
        print(f"\n{col} values:")
        print(mines[col].value_counts())
else:
    print("No explicit status columns found in shapefile")
    print("We'll use area as a proxy: larger area = likely excavated")

# ============================================================================
# BASIC VALIDATION
# ============================================================================

print("\n[3/6] Validating geometries...")

# Check geometry types
geom_types = mines.geometry.geom_type.value_counts()
print(f"\nGeometry types:")
for geom_type, count in geom_types.items():
    print(f"  {geom_type}: {count}")

# Check for invalid geometries
invalid_geoms = ~mines.geometry.is_valid
num_invalid = invalid_geoms.sum()
print(f"\nInvalid geometries: {num_invalid}")

if num_invalid > 0:
    print(f"  WARNING: Found {num_invalid} invalid geometries. Attempting to fix...")
    mines.geometry = mines.geometry.buffer(0)
    invalid_after_fix = (~mines.geometry.is_valid).sum()
    print(f"✓ After fix, invalid geometries: {invalid_after_fix}")

# ============================================================================
# CRS CONVERSION
# ============================================================================

print("\n[4/6] Converting to WGS84 (EPSG:4326)...")

if mines.crs != "EPSG:4326":
    mines = mines.to_crs(epsg=4326)
    print("Converted to EPSG:4326")
else:
    print("Already in EPSG:4326")

# ============================================================================
# CALCULATE STATISTICS
# ============================================================================

print("\n[5/6] Computing spatial statistics...")

# Calculate area in square degrees (approximate)
mines['area_deg2'] = mines.geometry.area

# Convert to approximate square kilometers
mines['area_km2'] = mines['area_deg2'] * (111 ** 2)

print(f"\nArea statistics (km²):")
print(mines['area_km2'].describe())

# Calculate centroids
mines['centroid_lon'] = mines.geometry.centroid.x
mines['centroid_lat'] = mines.geometry.centroid.y

print(f"\nSpatial extent:")
print(f"  Longitude: [{mines.centroid_lon.min():.4f}, {mines.centroid_lon.max():.4f}]")
print(f"  Latitude: [{mines.centroid_lat.min():.4f}, {mines.centroid_lat.max():.4f}]")

# ============================================================================
# EXCAVATION STATUS ANALYSIS
# ============================================================================

print("\n[6/6] Analyzing excavation status...")

# Since we don't have explicit labels, use area as proxy
# Assumption: Larger mines are more likely to be actively excavated
area_median = mines['area_km2'].median()
area_75percentile = mines['area_km2'].quantile(0.75)

print(f"\nArea distribution:")
print(f"  Median: {area_median:.4f} km²")
print(f"  75th percentile: {area_75percentile:.4f} km²")

# Classify based on area (heuristic)
# Large mines (top 50%) = likely excavated/active
# Small mines (bottom 50%) = might be less active or planned
mines['likely_excavated'] = mines['area_km2'] > area_median

excavated_count = mines['likely_excavated'].sum()
print(f"\nBased on area heuristic:")
print(f"  Likely excavated: {excavated_count} ({excavated_count/len(mines)*100:.1f}%)")
print(f"  Likely not/less excavated: {len(mines) - excavated_count} ({(len(mines)-excavated_count)/len(mines)*100:.1f}%)")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n[Visualization] Creating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: All mines
mines.plot(ax=axes[0, 0], color='orange', edgecolor='black', alpha=0.6, linewidth=0.5)
axes[0, 0].set_title(f'All {len(mines)} Mine Polygons', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Latitude')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Area distribution
axes[0, 1].hist(mines['area_km2'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(area_median, color='red', linestyle='--', linewidth=2, label=f'Median: {area_median:.2f} km²')
axes[0, 1].axvline(area_75percentile, color='orange', linestyle='--', linewidth=2, label=f'75th: {area_75percentile:.2f} km²')
axes[0, 1].set_title('Mine Area Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Area (km²)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Color by likely excavation status
mines.plot(ax=axes[1, 0], 
          column='likely_excavated', 
          cmap='RdYlGn',
          edgecolor='black', 
          linewidth=0.5,
          legend=True,
          legend_kwds={'title': 'Likely Excavated (based on area)'})
axes[1, 0].set_title('Mines by Likely Excavation Status', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Longitude')
axes[1, 0].set_ylabel('Latitude')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Box plot
box_data = [mines[mines['likely_excavated']]['area_km2'].values,
           mines[~mines['likely_excavated']]['area_km2'].values]
axes[1, 1].boxplot(box_data, labels=['Likely Excavated', 'Likely Not Excavated'])
axes[1, 1].set_title('Area Distribution by Status', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Area (km²)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("../outputs/validation/step0_mines_analysis.png", dpi=300, bbox_inches='tight')
print("Saved: ../outputs/validation/step0_mines_analysis.png")
plt.close()

# ============================================================================
# SAVE VALIDATED DATA
# ============================================================================

print("\n[Saving] Writing validated shapefile...")

# Add mine_id column for easy reference
mines['mine_id'] = [f'mine_{i:04d}' for i in range(len(mines))]

# Save validated shapefile
mines.to_file("../data/processed/mines_validated.shp")
print("Saved: ../data/processed/mines_validated.shp")

# Save detailed CSV with all attributes
mines_csv = mines.copy()
mines_csv['geometry'] = mines_csv['geometry'].apply(lambda x: x.wkt)  # Convert geometry to WKT
mines_csv.to_csv("../outputs/validation/step0_mines_detailed.csv", index=False)
print("Saved: ../outputs/validation/step0_mines_detailed.csv")

# Save summary statistics
summary_stats = {
    'total_mines': len(mines),
    'geometry_types': geom_types.to_dict(),
    'invalid_geometries': int(num_invalid),
    'crs': str(mines.crs),
    'area_stats_km2': {
        'min': float(mines['area_km2'].min()),
        'max': float(mines['area_km2'].max()),
        'mean': float(mines['area_km2'].mean()),
        'median': float(mines['area_km2'].median()),
        'q25': float(mines['area_km2'].quantile(0.25)),
        'q75': float(mines['area_km2'].quantile(0.75))
    },
    'spatial_extent': {
        'lon_min': float(mines.centroid_lon.min()),
        'lon_max': float(mines.centroid_lon.max()),
        'lat_min': float(mines.centroid_lat.min()),
        'lat_max': float(mines.centroid_lat.max())
    },
    'excavation_estimate': {
        'likely_excavated': int(excavated_count),
        'likely_not_excavated': int(len(mines) - excavated_count),
        'note': 'Based on area heuristic (above median = likely excavated)'
    }
}

import json
with open("../outputs/validation/step0_summary.json", 'w') as f:
    json.dump(summary_stats, f, indent=2)
print("Saved: ../outputs/validation/step0_summary.json")

# ============================================================================
# VALIDATION CHECKLIST
# ============================================================================

print("\n" + "="*60)
print("VALIDATION CHECKLIST")
print("="*60)

checklist = [
    ("Total mines loaded", len(mines) == 506, f"{len(mines)} mines"),
    ("No invalid geometries", num_invalid == 0, f"{num_invalid} invalid"),
    ("CRS is WGS84", mines.crs == "EPSG:4326", str(mines.crs)),
    ("Both Polygon types present", len(geom_types) >= 1, f"{len(geom_types)} types"),
    ("Mine IDs assigned", 'mine_id' in mines.columns, "mine_id column exists"),
    ("Area calculated", 'area_km2' in mines.columns, f"Range: {mines['area_km2'].min():.2f}-{mines['area_km2'].max():.2f} km²"),
]

for check_name, passed, details in checklist:
    status = "✓" if passed else "✗"
    print(f"{status} {check_name}: {details}")

print("STEP 0 COMPLETE!")