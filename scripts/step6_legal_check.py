"""
Check if excavation occurred in NO-GO mines
If excavation detected in a NO-GO mine → ILLEGAL VIOLATION
"""

import os
import numpy as np
import geopandas as gpd
import json
from tqdm import tqdm
import pandas as pd

OUTPUT_DIR = "../data/step6_violations"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("../outputs/validation", exist_ok=True)

print("="*60)
print("STEP 6: LEGAL INTERSECTION CHECK (CORRECTED)")
print("="*60)

# ============================================================================
# LOAD ZONE ASSIGNMENTS
# ============================================================================

print("\n[1/4] Loading zone assignments...")

# Load mines with zone labels
mines_with_zones = gpd.read_file("../data/processed/mines_with_zones.shp")

# Create dictionaries for quick lookup
nogo_mines = set(mines_with_zones[mines_with_zones['zone_type'] == 'nogo']['mine_id'])
legal_mines = set(mines_with_zones[mines_with_zones['zone_type'] == 'legal']['mine_id'])

print(f"NO-GO mines: {len(nogo_mines)}")
print(f"LEGAL mines: {len(legal_mines)}")

# ============================================================================
# CHECK EXCAVATION IN EACH MINE
# ============================================================================

print("\n[2/4] Checking for excavations in NO-GO zones...")

REGIONS_DIR = "../data/step5_regions"

# Load all region data
mine_region_files = [f for f in os.listdir(REGIONS_DIR) if f.endswith('_regions.json')]

violations = []
legal_excavations = []
no_activity = []

for region_file in tqdm(mine_region_files, desc="Checking mines"):
    mine_id = region_file.replace('_regions.json', '')
    
    # Load region data
    with open(os.path.join(REGIONS_DIR, region_file), 'r') as f:
        region_data = json.load(f)
    
    # Check if this mine has any excavation activity
    has_excavation = region_data.get('total_growth_m2', 0) > 0
    
    if not has_excavation:
        no_activity.append(mine_id)
        continue
    
    # Check if mine is in NO-GO zone
    if mine_id in nogo_mines:
        # VIOLATION! Excavation in NO-GO zone
        violations.append({
            'mine_id': mine_id,
            'violation_type': 'excavation_in_nogo_zone',
            'total_excavated_area_m2': region_data['total_growth_m2'],
            'max_area_m2': region_data['max_area_m2'],
            'growth_consistency': region_data['growth_consistency'],
            'num_time_periods': region_data['num_time_periods'],
            'severity': 'HIGH' if region_data['total_growth_m2'] > 10000 else 'MEDIUM'
        })
    
    elif mine_id in legal_mines:
        # Legal excavation in authorized zone
        legal_excavations.append({
            'mine_id': mine_id,
            'status': 'legal_excavation',
            'total_excavated_area_m2': region_data['total_growth_m2'],
            'max_area_m2': region_data['max_area_m2']
        })

print(f"\nAnalysis complete:")
print(f"  Violations (excavation in NO-GO): {len(violations)}")
print(f"  Legal excavations: {len(legal_excavations)}")
print(f"  No activity detected: {len(no_activity)}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[3/4] Saving violation data...")

# Save violations
with open(os.path.join(OUTPUT_DIR, 'violations.json'), 'w') as f:
    json.dump(violations, f, indent=2)

# Save legal excavations
with open(os.path.join(OUTPUT_DIR, 'legal_excavations.json'), 'w') as f:
    json.dump(legal_excavations, f, indent=2)

# Create violation summary DataFrame
if violations:
    df_violations = pd.DataFrame(violations)
    df_violations = df_violations.sort_values('total_excavated_area_m2', ascending=False)
    df_violations.to_csv(os.path.join(OUTPUT_DIR, 'violation_summary.csv'), index=False)
    
    print(f"\n{'='*60}")
    print("VIOLATION STATISTICS")
    print(f"{'='*60}")
    print(f"\nTotal violations: {len(df_violations)}")
    print(f"\nExcavated area in NO-GO zones:")
    print(df_violations['total_excavated_area_m2'].describe())
    print(f"\nSeverity breakdown:")
    print(df_violations['severity'].value_counts())
    print(f"\nTop 5 violators:")
    print(df_violations.head()[['mine_id', 'total_excavated_area_m2', 'severity']])
else:
    print("\nNo violations detected!")

# Create legal excavation summary
if legal_excavations:
    df_legal = pd.DataFrame(legal_excavations)
    df_legal.to_csv(os.path.join(OUTPUT_DIR, 'legal_excavations.csv'), index=False)
    print(f"\nLegal excavations: {len(df_legal)}")

print(f"\nSaved: {OUTPUT_DIR}/violations.json")
print(f"Saved: {OUTPUT_DIR}/violation_summary.csv")

# ============================================================================
# STATISTICS SUMMARY
# ============================================================================

print("\n[4/4] Generating summary statistics...")

summary = {
    'total_mines_analyzed': len(mine_region_files),
    'nogo_mines': len(nogo_mines),
    'legal_mines': len(legal_mines),
    'violations': {
        'count': len(violations),
        'percentage_of_nogo_mines': (len(violations) / len(nogo_mines) * 100) if nogo_mines else 0,
        'total_illegal_area_m2': sum(v['total_excavated_area_m2'] for v in violations)
    },
    'legal_activity': {
        'count': len(legal_excavations),
        'percentage_of_legal_mines': (len(legal_excavations) / len(legal_mines) * 100) if legal_mines else 0,
        'total_legal_area_m2': sum(le['total_excavated_area_m2'] for le in legal_excavations)
    },
    'no_activity': {
        'count': len(no_activity),
        'percentage': (len(no_activity) / len(mine_region_files) * 100) if mine_region_files else 0
    }
}

with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"\nTotal mines analyzed: {summary['total_mines_analyzed']}")
print(f"\nNO-GO zones:")
print(f"  Total NO-GO mines: {summary['nogo_mines']}")
print(f"  Violations detected: {summary['violations']['count']} ({summary['violations']['percentage_of_nogo_mines']:.1f}% of NO-GO mines)")
print(f"  Total illegal area: {summary['violations']['total_illegal_area_m2']:,.0f} m²")
print(f"\nLEGAL zones:")
print(f"  Total LEGAL mines: {summary['legal_mines']}")
print(f"  Legal excavations: {summary['legal_activity']['count']} ({summary['legal_activity']['percentage_of_legal_mines']:.1f}% of LEGAL mines)")
print(f"  Total legal area: {summary['legal_activity']['total_legal_area_m2']:,.0f} m²")
print(f"\nNo activity:")
print(f"  Mines with no excavation: {summary['no_activity']['count']} ({summary['no_activity']['percentage']:.1f}%)")

print("\n" + "="*60)
print("STEP 6 COMPLETE!")
print("="*60)
print(f"\nKey findings:")
if violations:
    print(f"  {len(violations)} violations detected in NO-GO zones!")
    print(f" {summary['violations']['total_illegal_area_m2']:,.0f} m² illegally excavated")
else:
    print(f"No violations detected")