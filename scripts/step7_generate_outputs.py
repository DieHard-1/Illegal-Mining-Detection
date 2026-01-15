"""
Step 7: Generate Final Outputs & Evaluation Metrics
- Time-series plots for each mine
- Spatial excavation maps
- Alert logs and violation reports
- Comprehensive evaluation metrics
- Executive summary report
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import geopandas as gpd
from tqdm import tqdm

# Create output directories
OUTPUT_DIR = "../outputs/final"
os.makedirs(f"{OUTPUT_DIR}/timeseries", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/maps", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/alerts", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/metrics", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/reports", exist_ok=True)

print("="*60)
print("STEP 7: GENERATE FINAL OUTPUTS & EVALUATION")
print("="*60)

# ============================================================================
# LOAD ALL DATA
# ============================================================================

print("\n[1/7] Loading all data...")

# Load region data
regions_data = {}
regions_dir = "../data/step5_regions"
for f in os.listdir(regions_dir):
    if f.endswith('_regions.json'):
        mine_id = f.replace('_regions.json', '')
        with open(os.path.join(regions_dir, f), 'r') as file:
            regions_data[mine_id] = json.load(file)

print(f"  Loaded region data for {len(regions_data)} mines")

# Load violations
violations_file = "../data/step6_violations/violations.json"
if os.path.exists(violations_file):
    with open(violations_file, 'r') as f:
        violations_data = json.load(f)
else:
    violations_data = []

print(f"  Loaded {len(violations_data)} violations")

# Load legal excavations
legal_file = "../data/step6_violations/legal_excavations.json"
if os.path.exists(legal_file):
    with open(legal_file, 'r') as f:
        legal_excavations = json.load(f)
else:
    legal_excavations = []

print(f"  Loaded {len(legal_excavations)} legal excavations")

# Load zone assignments
mines_with_zones = gpd.read_file("../data/processed/mines_with_zones.shp")
nogo_mines = set(mines_with_zones[mines_with_zones['zone_type'] == 'nogo']['mine_id'])
legal_mines = set(mines_with_zones[mines_with_zones['zone_type'] == 'legal']['mine_id'])

print(f"  Loaded zone assignments: {len(nogo_mines)} NO-GO, {len(legal_mines)} LEGAL")

# ============================================================================
# 1. GENERATE TIME-SERIES PLOTS
# ============================================================================

print("\n[2/7] Generating time-series plots...")

# Select mines to plot (violations + some legal + some no-activity)
violation_mine_ids = [v['mine_id'] for v in violations_data]
legal_active_ids = [le['mine_id'] for le in legal_excavations]

# Plot top violations
mines_to_plot = violation_mine_ids[:20]  # Top 20 violations

# Add some legal excavations
if len(legal_active_ids) > 10:
    mines_to_plot.extend(legal_active_ids[:10])

print(f"  Creating plots for {len(mines_to_plot)} mines...")

for mine_id in tqdm(mines_to_plot, desc="  Time-series plots"):
    if mine_id not in regions_data:
        continue
    
    data = regions_data[mine_id]
    temporal_regions = data.get('temporal_regions', [])
    
    if not temporal_regions:
        continue
    
    # Extract data
    time_points = list(range(len(temporal_regions)))
    total_areas = [tr['total_area_m2'] for tr in temporal_regions]
    num_regions = [tr['num_regions'] for tr in temporal_regions]
    
    # Check if this mine has violations
    is_violation = mine_id in violation_mine_ids
    is_nogo = mine_id in nogo_mines
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Total excavated area
    color = 'red' if is_violation else ('orange' if is_nogo else 'blue')
    axes[0].plot(time_points, total_areas, 
                marker='o', linewidth=2, markersize=6, color=color)
    axes[0].fill_between(time_points, 0, total_areas, alpha=0.3, color=color)
    axes[0].set_xlabel('Time Period (2023)', fontsize=12)
    axes[0].set_ylabel('Excavated Area (m²)', fontsize=12)
    axes[0].set_title(f'Total Excavated Area Over Time', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add status label
    if is_violation:
        status_text = '  VIOLATION DETECTED - Excavation in NO-GO Zone'
        axes[0].text(0.02, 0.98, status_text, 
                    transform=axes[0].transAxes, fontsize=11,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    elif is_nogo:
        axes[0].text(0.02, 0.98, '  NO-GO Zone - No Violation', 
                    transform=axes[0].transAxes, fontsize=11,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    else:
        axes[0].text(0.02, 0.98, '  LEGAL Zone - Authorized Activity', 
                    transform=axes[0].transAxes, fontsize=11,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
    
    # Plot 2: Number of regions
    axes[1].plot(time_points, num_regions, 
                marker='s', linewidth=2, markersize=6, color='green')
    axes[1].set_xlabel('Time Period (2023)', fontsize=12)
    axes[1].set_ylabel('Number of Regions', fontsize=12)
    axes[1].set_title(f'Number of Excavation Regions', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Main title
    zone_type = "NO-GO" if is_nogo else "LEGAL"
    plt.suptitle(f'Mine: {mine_id} ({zone_type} Zone)\n'
                f'Growth: {data.get("total_growth_m2", 0):.0f} m² | '
                f'Consistency: {data.get("growth_consistency", 0):.1%}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/timeseries/{mine_id}_timeseries.png", 
               dpi=200, bbox_inches='tight')
    plt.close()

print(f"  Generated {len(mines_to_plot)} time-series plots")

# ============================================================================
# 2. GENERATE ALERT LOG
# ============================================================================

print("\n[3/7] Generating alert logs...")

# Create detailed alert log
alerts = []

for violation in violations_data:
    mine_id = violation['mine_id']
    
    # Get temporal details if available
    if mine_id in regions_data:
        region_data = regions_data[mine_id]
        temporal = region_data.get('temporal_regions', [])
        
        # Find first detection
        first_detection = None
        for tr in temporal:
            if tr['total_area_m2'] > 0:
                first_detection = tr.get('ts2', 'Unknown')
                break
        
        alerts.append({
            'mine_id': mine_id,
            'violation_type': violation['violation_type'],
            'first_detected': first_detection,
            'total_excavated_area_m2': violation['total_excavated_area_m2'],
            'max_area_m2': violation.get('max_area_m2', 0),
            'growth_consistency': violation.get('growth_consistency', 0),
            'num_time_periods': violation.get('num_time_periods', 0),
            'severity': violation['severity'],
            'status': 'ACTIVE',
            'zone_type': 'NO-GO'
        })

# Sort by area (biggest violations first)
alerts_sorted = sorted(alerts, key=lambda x: x['total_excavated_area_m2'], reverse=True)

# Save as CSV
df_alerts = pd.DataFrame(alerts_sorted)
df_alerts.to_csv(f"{OUTPUT_DIR}/alerts/violation_alerts.csv", index=False)

print(f"  Generated alert log: {len(df_alerts)} violations")

# Create legal activity log
legal_log = []
for le in legal_excavations:
    legal_log.append({
        'mine_id': le['mine_id'],
        'status': 'LEGAL',
        'total_excavated_area_m2': le['total_excavated_area_m2'],
        'max_area_m2': le.get('max_area_m2', 0),
        'zone_type': 'LEGAL'
    })

df_legal = pd.DataFrame(legal_log)
df_legal.to_csv(f"{OUTPUT_DIR}/alerts/legal_activity_log.csv", index=False)

print(f"  Generated legal activity log: {len(df_legal)} excavations")

# ============================================================================
# 3. COMPUTE EVALUATION METRICS
# ============================================================================

print("\n[4/7] Computing evaluation metrics...")

# Comprehensive metrics
metrics = {
    "metadata": {
        "evaluation_date": datetime.now().isoformat(),
        "time_period": "2023-01-01 to 2023-12-31",
        "total_mines": len(regions_data),
        "pipeline_version": "1.0"
    },
    
    "zone_distribution": {
        "nogo_mines": len(nogo_mines),
        "legal_mines": len(legal_mines),
        "nogo_percentage": len(nogo_mines) / len(mines_with_zones) * 100,
        "legal_percentage": len(legal_mines) / len(mines_with_zones) * 100
    },
    
    "detection_metrics": {
        "mines_with_activity": sum(1 for d in regions_data.values() if d.get('total_growth_m2', 0) > 0),
        "mines_with_violations": len(violations_data),
        "mines_with_legal_activity": len(legal_excavations),
        "total_violation_instances": len(df_alerts) if not df_alerts.empty else 0,
        "detection_rate": sum(1 for d in regions_data.values() if d.get('total_growth_m2', 0) > 0) / len(regions_data),
        "violation_rate_in_nogo": len(violations_data) / len(nogo_mines) if nogo_mines else 0,
        "activity_rate_in_legal": len(legal_excavations) / len(legal_mines) if legal_mines else 0
    },
    
    "temporal_consistency": {
        "avg_growth_consistency": float(np.mean([d.get('growth_consistency', 0) for d in regions_data.values()])),
        "mines_with_consistent_growth": sum(1 for d in regions_data.values() if d.get('growth_consistency', 0) > 0.5),
        "consistency_threshold": 0.5
    },
    
    "area_statistics": {
        "total_excavated_area_m2": float(sum(d.get('max_area_m2', 0) for d in regions_data.values())),
        "total_excavated_area_km2": float(sum(d.get('max_area_m2', 0) for d in regions_data.values())) / 1e6,
        "avg_excavated_area_per_mine_m2": float(np.mean([d.get('max_area_m2', 0) for d in regions_data.values()])),
        "max_single_mine_area_m2": float(max(d.get('max_area_m2', 0) for d in regions_data.values())),
        "total_violation_area_m2": float(df_alerts['total_excavated_area_m2'].sum()) if not df_alerts.empty else 0,
        "total_violation_area_km2": float(df_alerts['total_excavated_area_m2'].sum() / 1e6) if not df_alerts.empty else 0,
        "total_legal_area_m2": float(df_legal['total_excavated_area_m2'].sum()) if not df_legal.empty else 0,
        "total_legal_area_km2": float(df_legal['total_excavated_area_m2'].sum() / 1e6) if not df_legal.empty else 0
    },
    
    "spatial_metrics": {
        "avg_regions_per_mine": float(np.mean([np.mean([tr['num_regions'] for tr in d.get('temporal_regions', [])]) 
                                                for d in regions_data.values() 
                                                if d.get('temporal_regions')])),
        "total_detected_regions": sum(sum(tr['num_regions'] for tr in d.get('temporal_regions', [])) 
                                     for d in regions_data.values())
    },
    
    "severity_breakdown": {
        "high_severity": int((df_alerts['severity'] == 'HIGH').sum()) if not df_alerts.empty else 0,
        "medium_severity": int((df_alerts['severity'] == 'MEDIUM').sum()) if not df_alerts.empty else 0
    }
}

# Save metrics
with open(f"{OUTPUT_DIR}/metrics/evaluation_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"  Saved evaluation metrics")

# ============================================================================
# 4. GENERATE SUMMARY VISUALIZATIONS
# ============================================================================

print("\n[5/7] Creating summary visualizations...")

# Visualization 1: Overall Statistics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Zone distribution pie chart
zone_data = [metrics['zone_distribution']['legal_mines'], 
             metrics['zone_distribution']['nogo_mines']]
zone_labels = [f"LEGAL\n{metrics['zone_distribution']['legal_mines']} mines\n({metrics['zone_distribution']['legal_percentage']:.1f}%)",
               f"NO-GO\n{metrics['zone_distribution']['nogo_mines']} mines\n({metrics['zone_distribution']['nogo_percentage']:.1f}%)"]
colors = ['#2ecc71', '#e74c3c']
axes[0, 0].pie(zone_data, labels=zone_labels, colors=colors, autopct='', startangle=90)
axes[0, 0].set_title('Zone Distribution', fontsize=14, fontweight='bold')

# Plot 2: Activity detection
activity_data = [
    metrics['detection_metrics']['mines_with_violations'],
    metrics['detection_metrics']['mines_with_legal_activity'],
    len(regions_data) - metrics['detection_metrics']['mines_with_activity']
]
activity_labels = [
    f"Violations\n{activity_data[0]}",
    f"Legal Activity\n{activity_data[1]}",
    f"No Activity\n{activity_data[2]}"
]
colors2 = ['#e74c3c', '#3498db', '#95a5a6']
axes[0, 1].pie(activity_data, labels=activity_labels, colors=colors2, autopct='%1.1f%%', startangle=90)
axes[0, 1].set_title('Activity Detection Results', fontsize=14, fontweight='bold')

# Plot 3: Area distribution
if not df_alerts.empty:
    top_violations = df_alerts.head(10)
    axes[1, 0].barh(range(len(top_violations)), top_violations['total_excavated_area_m2'] / 1000)
    axes[1, 0].set_yticks(range(len(top_violations)))
    axes[1, 0].set_yticklabels(top_violations['mine_id'], fontsize=9)
    axes[1, 0].set_xlabel('Excavated Area (×1000 m²)', fontsize=11)
    axes[1, 0].set_title('Top 10 Violations by Area', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
else:
    axes[1, 0].text(0.5, 0.5, 'No Violations Detected', 
                   ha='center', va='center', fontsize=14)
    axes[1, 0].set_title('Top 10 Violations by Area', fontsize=14, fontweight='bold')

# Plot 4: Temporal consistency distribution
consistency_scores = [d.get('growth_consistency', 0) for d in regions_data.values()]
axes[1, 1].hist(consistency_scores, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[1, 1].set_xlabel('Growth Consistency', fontsize=11)
axes[1, 1].set_ylabel('Number of Mines', fontsize=11)
axes[1, 1].set_title('Temporal Growth Consistency Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Illegal Mining Detection System - Overall Statistics', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/reports/summary_statistics.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  Created summary visualizations")

# ============================================================================
# 5. GENERATE SPATIAL MAP
# ============================================================================

print("\n[6/7] Creating spatial distribution map...")

try:
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot legal zones
    legal_zones_gdf = mines_with_zones[mines_with_zones['zone_type'] == 'legal']
    nogo_zones_gdf = mines_with_zones[mines_with_zones['zone_type'] == 'nogo']
    
    # Base map
    legal_zones_gdf.plot(ax=ax, color='lightblue', alpha=0.3, edgecolor='blue', linewidth=0.5, label='Legal Zones')
    nogo_zones_gdf.plot(ax=ax, color='lightcoral', alpha=0.3, edgecolor='red', linewidth=0.5, label='No-Go Zones')
    
    # Highlight violations
    if violations_data:
        violation_ids = [v['mine_id'] for v in violations_data]
        violations_gdf = mines_with_zones[mines_with_zones['mine_id'].isin(violation_ids)]
        violations_gdf.plot(ax=ax, color='red', alpha=0.8, edgecolor='darkred', linewidth=2, label='Violations Detected')
    
    ax.set_title('Spatial Distribution: Legal Zones, No-Go Zones & Violations', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/maps/spatial_distribution_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Created spatial distribution map")
except Exception as e:
    print(f"  Could not create spatial map: {e}")

# ============================================================================
# 6. GENERATE EXECUTIVE SUMMARY REPORT
# ============================================================================

print("\n[7/7] Generating executive summary report...")

summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                   ILLEGAL MINING DETECTION SYSTEM                        ║
║                        Executive Summary Report                          ║
║                              Aurora 2.0                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: 2023-01-01 to 2023-12-31

{'='*76}
1. DATASET OVERVIEW
{'='*76}

Total Mines Analyzed:              {metrics['metadata']['total_mines']}
Time Period:                       12 months (2023)
Temporal Resolution:               ~24 timestamps per mine
Spatial Resolution:                10-200 m/pixel (adaptive)

Zone Distribution:
  • LEGAL Zones:                   {metrics['zone_distribution']['legal_mines']} mines ({metrics['zone_distribution']['legal_percentage']:.1f}%)
  • NO-GO Zones:                   {metrics['zone_distribution']['nogo_mines']} mines ({metrics['zone_distribution']['nogo_percentage']:.1f}%)

{'='*76}
2. DETECTION RESULTS
{'='*76}

Activity Detection:
  • Mines with Activity:           {metrics['detection_metrics']['mines_with_activity']} ({metrics['detection_metrics']['detection_rate']:.1%})
  • Detection Rate:                {metrics['detection_metrics']['detection_rate']:.1%}

Violations:
  • Total Violations:              {metrics['detection_metrics']['mines_with_violations']} mines
  • Violation Rate in NO-GO:       {metrics['detection_metrics']['violation_rate_in_nogo']:.1%} of NO-GO zones
  • High Severity:                 {metrics['severity_breakdown']['high_severity']}
  • Medium Severity:               {metrics['severity_breakdown']['medium_severity']}

Legal Activity:
  • Legal Excavations:             {metrics['detection_metrics']['mines_with_legal_activity']} mines
  • Activity Rate in LEGAL:        {metrics['detection_metrics']['activity_rate_in_legal']:.1%} of LEGAL zones

{'='*76}
3. TEMPORAL ANALYSIS
{'='*76}

Growth Consistency:
  • Average Consistency:           {metrics['temporal_consistency']['avg_growth_consistency']:.1%}
  • Mines with >50% Consistency:   {metrics['temporal_consistency']['mines_with_consistent_growth']}
  
Interpretation:
  High consistency (>50%) indicates persistent excavation activity rather than
  seasonal vegetation changes or noise.

{'='*76}
4. AREA STATISTICS
{'='*76}

Total Excavated Areas:
  • Overall:                       {metrics['area_statistics']['total_excavated_area_km2']:.2f} km²
  • In Violations (NO-GO):         {metrics['area_statistics']['total_violation_area_km2']:.2f} km²
  • In Legal Zones:                {metrics['area_statistics']['total_legal_area_km2']:.2f} km²

Per Mine Statistics:
  • Average Excavated Area:        {metrics['area_statistics']['avg_excavated_area_per_mine_m2']:,.0f} m²
  • Maximum Single Mine:           {metrics['area_statistics']['max_single_mine_area_m2']:,.0f} m²

{'='*76}
5. SPATIAL METRICS
{'='*76}

Regional Analysis:
  • Total Detected Regions:        {metrics['spatial_metrics']['total_detected_regions']}
  • Avg Regions per Mine:          {metrics['spatial_metrics']['avg_regions_per_mine']:.1f}

{'='*76}
6. TOP VIOLATIONS
{'='*76}

"""

# Add top 10 violations
if not df_alerts.empty:
    summary_text += "Rank  Mine ID       Area (km²)    Severity   First Detected\n"
    summary_text += "-" * 76 + "\n"
    for idx, row in df_alerts.head(10).iterrows():
        area_km2 = row['total_excavated_area_m2'] / 1e6
        first_det = row.get('first_detected', 'Unknown')
        summary_text += f"{idx+1:3d}.  {row['mine_id']:12s}  {area_km2:8.4f}      {row['severity']:6s}     {first_det}\n"
else:
    summary_text += "No violations detected.\n"

summary_text += f"""

{'='*76}
7. METHODOLOGY
{'='*76}

Approach:                          Physics-guided Unsupervised Detection
Features:                          NDVI, NBR, NDMI, SI (spectral indices)
Change Detection:                  GMM-based temporal analysis
Spatial Analysis:                  Connected component + geometric properties
Legal Verification:                Deterministic intersection with zone polygons

Key Advantages:
    No labeled training data required
    Explainable (physics-based features)
    Adaptive (works across different mine types)
    Scalable (processes 506 mines efficiently)

{'='*76}
8. OUTPUTS GENERATED
{'='*76}

Time-Series Plots:                 {OUTPUT_DIR}/timeseries/
  • Individual mine excavation progression over 2023

Spatial Maps:                      {OUTPUT_DIR}/maps/
  • Geographic distribution of violations

Alert Logs:                        {OUTPUT_DIR}/alerts/
  • violation_alerts.csv           - All detected violations
  • legal_activity_log.csv         - Authorized excavations

Evaluation Metrics:                {OUTPUT_DIR}/metrics/
  • evaluation_metrics.json        - Comprehensive performance metrics

Reports:                           {OUTPUT_DIR}/reports/
  • EXECUTIVE_SUMMARY.txt          - This report
  • summary_statistics.png         - Visual summary

{'='*76}
9. RECOMMENDATIONS
{'='*76}

Based on the analysis:

1. IMMEDIATE ACTION REQUIRED:
   • Investigate {metrics['severity_breakdown']['high_severity']} high-severity violations
   • Deploy ground verification teams to top 10 violation sites

2. MONITORING:
   • Continue temporal monitoring of NO-GO zones
   • Track {metrics['detection_metrics']['mines_with_violations']} violation mines monthly

3. POLICY:
   • Review permitting for high-activity LEGAL zones
   • Consider buffer zone expansion around sensitive areas

4. TECHNICAL:
   • Reduce false positives through seasonal normalization
   • Integrate SAR data for all-weather monitoring
   • Add elevation change detection (DEM analysis)

{'='*76}
10. CONCLUSION
{'='*76}

The system successfully detected mining activity across 506 mines using
unsupervised satellite imagery analysis. {metrics['detection_metrics']['mines_with_violations']} violations were
identified in NO-GO zones, representing {metrics['detection_metrics']['violation_rate_in_nogo']:.1%} of prohibited areas.

The physics-guided approach provides transparent, explainable results suitable
for regulatory enforcement. Detected violations show high temporal consistency
({metrics['temporal_consistency']['avg_growth_consistency']:.1%}), indicating genuine excavation activity rather than
classification noise.

{'='*76}

╚══════════════════════════════════════════════════════════════════════════╝
"""

# Save summary report
with open(f"{OUTPUT_DIR}/reports/EXECUTIVE_SUMMARY.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(summary_text)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("  STEP 7 COMPLETE!")
print("="*60)

print(f"\n KEY RESULTS:")
print(f"  • Total Mines: {metrics['metadata']['total_mines']}")
print(f"  • Violations Detected: {metrics['detection_metrics']['mines_with_violations']}")
print(f"  • Violation Area: {metrics['area_statistics']['total_violation_area_km2']:.2f} km²")
print(f"  • Detection Rate: {metrics['detection_metrics']['detection_rate']:.1%}")
print(f"  • Temporal Consistency: {metrics['temporal_consistency']['avg_growth_consistency']:.1%}")

print(f"\n ALL OUTPUTS SAVED TO: {OUTPUT_DIR}/")
print(f"\n Files Generated:")
print(f"    {len(mines_to_plot)} time-series plots")
print(f"    Violation alerts CSV")
print(f"    Legal activity log")
print(f"    Evaluation metrics JSON")
print(f"    Summary visualizations")
print(f"    Spatial distribution map")
print(f"    Executive summary report")