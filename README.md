# UAE High-Resolution Climate Downscaling using CNN-LSTM with Fuzzy Physics

A deep learning pipeline for downscaling coarse Global Climate Model (GCM) outputs to high-resolution 17×17 spatial grids over the United Arab Emirates, with a physics-informed Urban Heat Island correction layer.

This project was developed as part of our undergraduate thesis at Birla Institute of Technology and Science (BITS Pilani, Dubai Campus).

---

## What does this project do?

GCMs like MIROC6 and MPI-ESM produce climate projections at very coarse spatial resolutions (typically 100-250 km grid cells). For a country as small as the UAE, this means the entire nation gets maybe 4-9 grid points — nowhere near enough to study localized phenomena like urban heat islands in Dubai or coastal humidity gradients near Abu Dhabi.

We built a CNN-LSTM neural network that learns to spatially downscale these coarse grids (3×3 or 2×2) into fine-grained 17×17 grids (~25 km resolution) for five meteorological variables simultaneously:

- **T_avg** — Daily average temperature (°C)
- **PCP** — Precipitation (mm)
- **AP** — Atmospheric pressure (hPa)
- **RH** — Relative humidity (%)
- **WS** — Wind speed (m/s)

After the neural network generates its predictions, a fuzzy logic engine applies physically motivated corrections for urban effects — adding UHI warming to temperature, reducing humidity (urban dry island), and dampening wind speed (surface roughness) — based on the spatial proximity of each pixel to known urban zones.

## Architecture

The model uses a two-stage **Transposed Convolution** block to learnable-upsample the coarse GCM input, followed by a 3-layer CNN with residual connections for spatial feature extraction, and a ConvLSTM cell for temporal sequence modeling across a 14-day sliding window.

```
Input (B, 14, 5, 3, 3)
    │
    ├── ConvTranspose2d: 3×3 → 9×9
    ├── ConvTranspose2d: 9×9 → 17×17
    │
    ├── Conv2d + BatchNorm + ReLU (×3, with residual skip)
    │
    ├── ConvLSTM Cell (hidden_dim=64)
    │
    └── Conv2d 1×1 → Output (B, 14, 5, 17, 17)
```

Total trainable parameters: ~379K

## Project Structure

```
├── models.py                  # CNN-LSTM architecture definition
├── data_loader.py             # Dataset class, normalization, log-transform for PCP
├── train.py                   # Training loop with channel-weighted Huber loss
├── inference.py               # Historical inference (1950-2014) + UHI + Mann-Kendall
├── project_future.py          # Future SSP245/SSP585 projections (2015-2100)
├── fuzzy_uhi.py               # Fuzzy logic Urban Heat Island correction engine
├── statistical_analysis.py    # TFPW-MK, SQ-MK trend detection, WMO 30-year analysis
│
├── evaluate_accuracy.py       # Basic spatial accuracy vs ERA5
├── ablation_study.py          # Raw vs Fuzzy ablation + Wilcoxon significance test
├── publication_evaluation.py  # Taylor diagram, per-pixel R², seasonality, extremes
├── baseline_comparison.py     # CNN-LSTM vs bilinear/bicubic interpolation
│
├── export_simplemaps.py       # Converts predictions to SimpleMaps mapdata.js
├── plot_dec29_normalization.py # December 29th spatial normalization study
│
├── MIROC6.py                  # Data fetching script for MIROC6 from CMIP6
├── MPI-ESM1-2-LR.py          # Data fetching script for MPI-ESM from CMIP6
├── Historical Baseline.py     # ERA5 data fetching via CDS API
├── fetch_future_miroc6.py     # Fetch SSP scenario projections
├── validate_datasets.py       # Sanity checks on all CSV inputs
│
├── requirements.txt
└── README.md
```

## Data

All training and evaluation data is sourced from:

- **ERA5 Reanalysis** (ECMWF via CDS API) — used as ground truth at 0.25° resolution, regridded to 17×17 over the UAE
- **MIROC6** (CMIP6) — 3×3 native grid over the UAE
- **MPI-ESM1-2-LR** (CMIP6) — 2×2 native grid over the UAE

Training period: 1950–2014 (historical). Validation: 2015–2025 (unseen). Projection: 2015–2100 (SSP2-4.5 and SSP5-8.5).

The CSV files are not included in this repository due to their size (~700 MB total). They can be regenerated using the fetch scripts (`MIROC6.py`, `MPI-ESM1-2-LR.py`, `Historical Baseline.py`, `fetch_future_miroc6.py`).

## Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU for training (CPU works but is very slow).

## Usage

### Training

```bash
python train.py
```

Trains separate downscaler models for each configured GCM (MIROC6 and MPI-ESM) against the ERA5 ground truth. Checkpoints are saved as `best_downscaler_{GCM_NAME}.pth`. The training uses:

- Channel-weighted Huber loss (PCP ×5, WS ×2, RH ×1.5) to prevent the model from ignoring difficult variables
- Log-transform on precipitation before normalization to handle the zero-inflated desert rainfall distribution
- ReduceLROnPlateau scheduler + early stopping (patience=7)

### Inference & Projections

```bash
python inference.py            # Historical 1950-2014
python project_future.py       # Future SSP scenarios 2015-2100
```

Both scripts apply the fuzzy UHI correction after neural network inference and enforce physical bounds (PCP ≥ 0, 0 ≤ RH ≤ 100, WS ≥ 0).

### Evaluation

```bash
python evaluate_accuracy.py        # Spatial accuracy maps
python ablation_study.py           # Raw vs Fuzzy + statistical significance
python publication_evaluation.py   # Taylor diagram, R² maps, extremes
python baseline_comparison.py      # vs bilinear/bicubic interpolation
```

### Additional Analysis

```bash
python export_simplemaps.py        # Generate mapdata.js for web visualization
python plot_dec29_normalization.py  # Dec 29th spatial warming study
```

## Results

Performance on the unseen 2015-2025 validation window:

| Variable | MAE | RMSE | R² | Improvement over Bilinear |
|----------|-----|------|-----|--------------------------|
| Temperature | 1.67°C | 2.22°C | 0.878 | 78% |
| Precipitation | 0.18 mm | 1.74 mm | 0.001 | 59% |
| Air Pressure | 2.09 hPa | 2.67 hPa | 0.955 | 74% |
| Rel. Humidity | 8.76% | 11.00% | 0.656 | 77% |
| Wind Speed | 1.30 m/s | 1.82 m/s | -0.226 | 24% |

The Wilcoxon signed-rank test confirms that the fuzzy physics correction produces a statistically significant improvement (p < 0.05) for temperature, humidity, and wind speed.

Under SSP5-8.5, the model projects warming rates of 0.018°C/yr (2015-2044) accelerating to 0.035°C/yr (2075-2100) for the UAE, with MPI-ESM detecting a potential mutation point at 2074.

## Notes

- Precipitation R² is expected to be low — daily spatial rainfall in the UAE is inherently stochastic and zero-inflated. The model correctly captures the volume and frequency distributions, which the SQ-MK trend test validates.
- Wind speed R² is negative because the fuzzy engine applies aggressive surface roughness corrections over urban areas, which trades pixel-level accuracy for physical realism.
- The model adapts to different GCM input resolutions automatically. MIROC6 (3×3) uses the ConvTranspose2d path directly; MPI-ESM (2×2) adds a lightweight bilinear resize to match the 17×17 target.


## License

This project was developed for academic purposes at BITS Pilani, Dubai Campus.
