"""
Step 1: Sentinel-2 Data Fetching
- Fetch Sentinel-2 imagery for each mine polygon
- Time period: 2023 only (12 months)
- 2 samples per month = 24 timestamps per mine
- Output: Raw bands (B04, B08, B11, B12, SCL) as .npy files
- Use fixed size instead of resolution to avoid large mine issues
"""

import os
import json
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
from tqdm import tqdm

from sentinelhub import (
    SHConfig, BBox, CRS,
    SentinelHubRequest, DataCollection,
    MimeType
)

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = "2023-01-01"
END_DATE   = "2023-12-31"
SAMPLES_PER_MONTH = 2  # 2 samples/month = 24 timestamps/year

# 500m Buffer around mine polygon (in meters)
BUFFER_METERS = 500 

# Instead of specifying resolution, we specify max image dimensions
MAX_IMAGE_SIZE = 512  # Max 512x512 pixels per image
# This means each pixel will be: (bbox_width / 512) meters

# Output directory
OUTPUT_DIR = "../data/step1_sentinel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SENTINEL HUB CREDENTIALS
# ============================================================================

# Set up credentials
config = SHConfig()

# Load from environment variables
import os as env_os
if env_os.getenv('SH_CLIENT_ID'):
    config.sh_client_id = env_os.getenv('SH_CLIENT_ID')
    config.sh_client_secret = env_os.getenv('SH_CLIENT_SECRET')
    print(" Loaded credentials from environment variables")
else:
    print("  WARNING: No credentials found!")
    print("   Set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables")

# ============================================================================
# LOAD MINES
# ============================================================================

print("="*60)
print("STEP 1: SENTINEL-2 DATA FETCHING")
print("="*60)

print("\n[1/4] Loading mine polygons...")
mines = gpd.read_file("../data/processed/mines_validated.shp")
mines = mines.to_crs(epsg=4326)

print(f" Loaded {len(mines)} mine polygons")

# ============================================================================
# TEST MODE: Process only first N mines for testing
# ============================================================================

TEST_MODE = False  # Set to False to process all mines
NUM_TEST_MINES = 5

if TEST_MODE:
    print(f"\n  TEST MODE: Processing only {NUM_TEST_MINES} mines")
    mines = mines.head(NUM_TEST_MINES)
    print(f"   Set TEST_MODE=False in the script to process all mines")

# ============================================================================
# TIME GRID CREATION
# ============================================================================

def generate_time_intervals(start, end, samples_per_month):
    """
    Generate time intervals for 2023
    
    Returns list of (start_date, end_date) tuples
    Each tuple represents a 1-day window for cloud-free selection
    """
    dates = []
    current = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    while current <= end_dt:
        # Get month boundaries
        month_start = current.replace(day=1)
        
        # Calculate next month
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1, day=1) - timedelta(days=1)
        
        # Calculate step size within month
        days_in_month = (month_end - month_start).days + 1
        step_days = days_in_month // samples_per_month

        for i in range(samples_per_month):
            t0 = month_start + timedelta(days=i * step_days)
            t1 = t0 + timedelta(days=7)  # 7-day window for cloud-free selection
            
            if t0 <= end_dt:
                dates.append((t0.date().isoformat(), t1.date().isoformat()))

        # Move to next month
        current = month_end + timedelta(days=1)

    return dates

time_intervals = generate_time_intervals(START_DATE, END_DATE, SAMPLES_PER_MONTH)

print(f"\n[2/4] Generated {len(time_intervals)} temporal samples for 2023")
print(f"   First sample: {time_intervals[0]}")
print(f"   Last sample: {time_intervals[-1]}")

# ============================================================================
# EVALSCRIPT: WHAT DATA TO FETCH
# ============================================================================

# This script tells Sentinel Hub what to return
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "B11", "B12", "SCL"],
    output: { 
      bands: 5,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  // Mask clouds and cloud shadows
  // SCL values: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus
  if (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9 || sample.SCL == 10) {
    return [NaN, NaN, NaN, NaN, sample.SCL];
  }
  
  // Return reflectance values (0-1 range, already in L2A)
  return [sample.B04, sample.B08, sample.B11, sample.B12, sample.SCL];
}
"""

# ============================================================================
# CALCULATE APPROPRIATE IMAGE SIZE FOR BBOX
# ============================================================================

def calculate_image_size(bbox, max_size=512):
    """
    Calculate appropriate image size maintaining aspect ratio
    
    Args:
        bbox: BBox object or tuple (minx, miny, maxx, maxy)
        max_size: Maximum dimension (width or height)
    
    Returns:
        (width, height) tuple in pixels
    """
    # Extract coordinates from BBox object
    if hasattr(bbox, 'lower_left') and hasattr(bbox, 'upper_right'):
        minx, miny = bbox.lower_left
        maxx, maxy = bbox.upper_right
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        minx, miny, maxx, maxy = bbox
    else:
        # Fallback: try to get geometry
        minx, miny, maxx, maxy = bbox.geometry.bounds
    
    # Calculate aspect ratio
    width_deg = maxx - minx
    height_deg = maxy - miny
    aspect_ratio = width_deg / height_deg
    
    # Determine dimensions maintaining aspect ratio
    if aspect_ratio > 1:
        # Wider than tall
        width = max_size
        height = int(max_size / aspect_ratio)
    else:
        # Taller than wide
        height = max_size
        width = int(max_size * aspect_ratio)
    
    # Ensure minimum size of 64 pixels
    width = max(64, width)
    height = max(64, height)
    
    return (width, height)

# ============================================================================
# MAIN FETCHING LOOP
# ============================================================================

print("\n[3/4] Fetching Sentinel-2 data...")
print(f"   Total requests: {len(mines)} mines × {len(time_intervals)} timestamps = {len(mines) * len(time_intervals)}")
print(f"   Image size: Up to {MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE} pixels per mine")
print(f"   Estimated time: ~{len(mines) * len(time_intervals) * 2 / 60:.0f} minutes")

failed_downloads = []
successful_downloads = 0

for idx, row in tqdm(mines.iterrows(), total=len(mines), desc="Processing mines"):
    mine_id = row['mine_id']
    mine_dir = os.path.join(OUTPUT_DIR, mine_id)
    os.makedirs(mine_dir, exist_ok=True)

    # Get geometry and create buffered bounding box
    geom = row.geometry
    
    # Buffer: convert meters to degrees (approximate at equator: 1° ≈ 111km)
    buffer_deg = BUFFER_METERS / 111000  # ~0.0045 degrees for 500m
    geom_buffered = geom.buffer(buffer_deg)
    
    # Create bounding box
    bbox = BBox(bbox=geom_buffered.bounds, crs=CRS.WGS84)
    
    # Get bbox coordinates for calculation
    bbox_coords = geom_buffered.bounds  # (minx, miny, maxx, maxy)
    
    # Calculate appropriate image size for this bbox
    img_width, img_height = calculate_image_size(bbox_coords, max_size=MAX_IMAGE_SIZE)
    
    # Calculate actual resolution
    minx, miny, maxx, maxy = bbox_coords
    actual_resolution_x = ((maxx - minx) * 111000) / img_width  # meters per pixel
    actual_resolution_y = ((maxy - miny) * 111000) / img_height
    avg_resolution = (actual_resolution_x + actual_resolution_y) / 2

    # Save metadata
    metadata = {
        "mine_id": mine_id,
        "bbox": list(bbox_coords),  # Use the tuple we already have
        "bbox_explanation": "minx, miny, maxx, maxy in WGS84",
        "buffer_meters": BUFFER_METERS,
        "image_size": [img_width, img_height],
        "actual_resolution_m": round(avg_resolution, 2),
        "num_timestamps": len(time_intervals),
        "year": 2023
    }

    # Fetch each timestamp
    timestamps_fetched = 0
    
    for t_start, t_end in time_intervals:
        filename = f"{t_start}.npy"
        out_path = os.path.join(mine_dir, filename)

        # Skip if already downloaded
        if os.path.exists(out_path):
            timestamps_fetched += 1
            continue

        try:
            # Create request with FIXED SIZE (not resolution)
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(t_start, t_end),
                        mosaicking_order="leastCC"  # Least cloud coverage
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=bbox,
                size=(img_width, img_height),  # FIXED SIZE instead of resolution!
                config=config,
            )

            # Download
            data = request.get_data()[0]
            
            # Save as numpy array
            # Shape: (height, width, 5) where 5 = [B04, B08, B11, B12, SCL]
            np.save(out_path, data)
            timestamps_fetched += 1
            successful_downloads += 1
            
        except Exception as e:
            error_msg = str(e)
            failed_downloads.append({
                'mine_id': mine_id,
                'timestamp': t_start,
                'error': error_msg,
                'bbox_size_deg': f"{maxx-minx:.4f} x {maxy-miny:.4f}",
                'attempted_resolution': round(avg_resolution, 2)
            })
            continue

    # Update metadata with actual fetched count
    metadata['timestamps_fetched'] = timestamps_fetched
    
    # Save metadata after all timestamps processed
    with open(os.path.join(mine_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

# ============================================================================
# VALIDATION
# ============================================================================

print("\n[4/4] Validating downloads...")

validation_stats = []

for mine_id in mines['mine_id']:
    mine_path = os.path.join(OUTPUT_DIR, mine_id)
    
    if not os.path.exists(mine_path):
        continue
    
    npy_files = [f for f in os.listdir(mine_path) if f.endswith('.npy')]
    
    if npy_files:
        # Load one sample to check dimensions
        sample_path = os.path.join(mine_path, npy_files[0])
        sample_data = np.load(sample_path)
        
        height, width, bands = sample_data.shape
        
        # Calculate cloud coverage (NaN percentage)
        nan_percent = np.isnan(sample_data[:, :, 0]).sum() / (height * width) * 100
        
        # Load metadata
        meta_path = os.path.join(mine_path, 'metadata.json')
        actual_resolution = 0
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                actual_resolution = meta.get('actual_resolution_m', 0)
        
        validation_stats.append({
            'mine_id': mine_id,
            'num_timestamps': len(npy_files),
            'expected_timestamps': len(time_intervals),
            'height_pixels': height,
            'width_pixels': width,
            'bands': bands,
            'sample_cloud_pct': nan_percent,
            'actual_resolution_m': actual_resolution
        })

# Save validation report
import pandas as pd
df_validation = pd.DataFrame(validation_stats)

print(f"\n{'='*60}")
print("VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"Mines processed: {len(df_validation)}/{len(mines)}")
print(f"Successful downloads: {successful_downloads}")
print(f"Failed downloads: {len(failed_downloads)}")

if not df_validation.empty:
    print(f"\nTimestamps per mine:")
    print(df_validation['num_timestamps'].describe())
    print(f"\nImage dimensions:")
    print(f"  Height: {df_validation['height_pixels'].mean():.0f} ± {df_validation['height_pixels'].std():.0f} pixels")
    print(f"  Width: {df_validation['width_pixels'].mean():.0f} ± {df_validation['width_pixels'].std():.0f} pixels")
    print(f"\nActual resolution:")
    print(f"  Min: {df_validation['actual_resolution_m'].min():.1f} m/pixel")
    print(f"  Max: {df_validation['actual_resolution_m'].max():.1f} m/pixel")
    print(f"  Mean: {df_validation['actual_resolution_m'].mean():.1f} m/pixel")
    print(f"\nAverage cloud coverage: {df_validation['sample_cloud_pct'].mean():.2f}%")

if failed_downloads:
    print(f"\n  Failed downloads: {len(failed_downloads)}")
    # Save failed downloads for debugging
    with open("../outputs/validation/step1_failed_downloads.json", 'w') as f:
        json.dump(failed_downloads, f, indent=2)
    print("Details saved to: ../outputs/validation/step1_failed_downloads.json")

df_validation.to_csv("../outputs/validation/step1_download_stats.csv", index=False)
print("\n✓ Saved: ../outputs/validation/step1_download_stats.csv")

print(f"\n{'='*60}")
print("✓ STEP 1 COMPLETE!")
print(f"{'='*60}")
print(f"\nData saved to: {OUTPUT_DIR}/")
print(f"Each mine has ~{len(time_intervals)} timestamps (2023 only)")
print(f"Each timestamp is a numpy array: (H × W × 5)")
print(f"  - Bands: [B04(Red), B08(NIR), B11(SWIR1), B12(SWIR2), SCL(cloud mask)]")
print(f"  - Image size: Adaptive (max {MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE} pixels)")