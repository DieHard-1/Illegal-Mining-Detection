dd## 1. Project Overview

This project implements an end-to-end, unsupervised system for detecting illegal mining activity using multi-temporal Sentinel-2 satellite imagery.

The system analyzes **506 mining sites** and detects excavation by modeling mining as a **persistent physical land transformation**, rather than as a transient spectral anomaly. Pixel-level change detection, spatial aggregation, temporal growth validation, and regulatory zone checks are combined to produce explainable and auditable alerts.

No labeled training data is used. All decisions are physics-guided and rule-based.

---

## 2. Core Concept

Mining activity causes irreversible land-cover changes such as vegetation removal, soil exposure, and surface disturbance. These effects are detected through:

- Persistent drops in vegetation indices (NDVI)
- Changes in disturbance-related indices (NBR, NDMI, Soil Index)
- Spatially coherent disturbed regions
- Monotonic growth of excavated area over time

Early pipeline stages allow noisy detections, while later stages enforce strict spatial and temporal constraints before confirming excavation or illegality.

---

## 3. Directory Structure

```text
Illegal-Mining-Detection/
│
├── data/
│ ├── raw/
│ │ └── mines_cil_polygon/
│ │
│ ├── processed/
│ │
│ ├── step1_sentinel/
│ │ └── mine_xxxx/
│ │ ├── YYYY-MM-DD.npy
│ │ └── metadata.json
│ │
│ ├── step3_indices/
│ │ └── mine_xxxx/
│ │
│ ├── step4_changes/
│ │ └── mine_xxxx/
│ │ └── change_*.npz
│ │
│ └── step5_regions/
│ └── mine_xxxx_regions.json
│
├── outputs/
│ ├── validation/
│ │ └── step2_zone_assignments.csv
│ │
│ └── final/
│ ├── alerts/
│ │ └── violation_alerts.csv
│ │
│ ├── metrics/
│ │ └── evaluation_metrics.json
│ │
│ └── reports/
│ └── summary_statistics.png
│
├── scripts/
│ ├── step0_validate_mines.py
│ ├── step1_sentinel_fetch.py
│ ├── step2_create_zones.py
│ ├── step3_compute_indices.py
│ ├── step4_change_detection.py
│ ├── step5_spatial_analysis.py
│ └── step6_legal_check.py
│
├── web/
│ └── app.py
│
└── README.md
```
Please download and extract the data directory and place in the directory according to the file structure given below [link](https://drive.google.com/file/d/1DWoU4oEyWLpueobO55T86ewA9aLhmnl7/view?usp=sharing)

---

## 4. Pipeline Description

### Step 0: Mine Validation  
`scripts/step0_validate_mines.py`

- Validates mine polygons
- Removes degenerate or invalid geometries
- Uses polygon area as a proxy to filter noise

Output: validated mine set

---

### Step 1: Sentinel-2 Data Sampling  
`scripts/step1_sentinel_fetch.py`

- Fetches Sentinel-2 imagery for each mine and timestamp
- Each mine is represented as a 512 × 512 pixel tile
- Identical spatial grids are used across time
- Band values are normalized consistently across the pipeline

Output:  
`data/step1_sentinel/mine_xxxx/YYYY-MM-DD.npy`

---

### Step 2: Zone Assignment  
`scripts/step2_create_zones.py`

- Assigns each mine to a regulatory zone:
  - LEGAL
  - NO-GO
- Used later for legality checks

Output:  
`outputs/validation/step2_zone_assignments.csv`

---

### Step 3: Spectral Index Computation  
`scripts/step3_compute_indices.py`

Computed per pixel:
- NDVI (vegetation loss)
- NBR (surface disturbance)
- NDMI (moisture change)
- Soil Index

Output:  
`data/step3_indices/mine_xxxx/`

---

### Step 4: Change Detection (Pixel Level)  
`scripts/step4_change_detection.py`

- Compares consecutive timestamps
- Flags pixels with physically meaningful spectral change
- Produces excavation candidate masks
- This stage is intentionally recall-oriented and noisy

Fluctuations at this stage are expected.

Output:  
`data/step4_changes/mine_xxxx/change_*.npz`

---

### Step 5: Spatial and Temporal Aggregation  
`scripts/step5_spatial_analysis.py`

- Removes isolated noisy pixels
- Enforces spatial coherence using connected components
- Tracks excavated area over time
- Applies a monotonic envelope: excavated area may grow but never shrink

Computed metrics:
- Total excavated area
- Maximum pit area
- Growth consistency
- Temporal region evolution

Output:  
`data/step5_regions/mine_xxxx_regions.json`

---

### Step 6: Legal Compliance Check  
`scripts/step6_legal_check.py`

- Cross-checks confirmed excavation against zone type
- Declares violations only when:
  - Excavation exists
  - Growth is persistent
  - Mine lies in a NO-GO zone
- Records first violation event and severity

Outputs:
- `outputs/final/alerts/violation_alerts.csv`
- `outputs/final/metrics/evaluation_metrics.json`

---

## 5. Dashboard Application

Location:  
`web/app.py`

Run using:

```bash
cd web
streamlit run app.py



