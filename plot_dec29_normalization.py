"""
plot_dec29_normalization.py - December 29th Maximum Temperature Normalization Study
===================================================================================
Isolates December 29th from the future projection cubes and analyzes how the 
spatial distribution of daily maximum temperatures shifts across climate epochs.

Method:
    1. Extract Dec 29 temperature grids for years 2030, 2060, 2090
    2. For each grid, find T_min (the minimum of the max-temperature across all pixels)
    3. Normalize: T_normalized = T - T_min
    4. Plot the three normalized distributions side-by-side with consistent color scales
    
This normalization isolates the *relative spatial pattern* of warming, removing
the absolute baseline shift so we can see if urbanization hotspots intensify
independently of the regional warming signal.
"""
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────
FEATURES = ['T_avg', 'PCP', 'AP', 'RH', 'WS']
NPY_PATH = 'results_future/ssp245/uhi_adjusted_full_ssp245.npy'
MIROC_CSV = 'MIROC6_UAE_SSP245_2015_2100.csv'
OUTPUT_DIR = 'evaluation/dec29_study'
TARGET_YEARS = [2030, 2060, 2090]
TARGET_MONTH = 12
TARGET_DAY = 29

# UAE 17x17 grid geographic axis labels
_LAT_TICKS = [0, 4, 8, 12, 16]
_LAT_LABELS = ['22.0N', '23.0N', '24.0N', '25.0N', '26.0N']
_LON_TICKS = [0, 4, 8, 12, 16]
_LON_LABELS = ['52.0E', '53.0E', '54.0E', '55.0E', '56.0E']

def apply_geo_axes(ax):
    """Apply geographic coordinate labels to a 17x17 heatmap axis."""
    ax.set_yticks(_LAT_TICKS)
    ax.set_yticklabels(_LAT_LABELS, fontsize=9)
    ax.set_xticks(_LON_TICKS)
    ax.set_xticklabels(_LON_LABELS, fontsize=9)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_xlabel('Longitude', fontsize=10)


def find_dec29_index(prediction_dates, year):
    """
    Find the index in prediction_dates that corresponds to December 29 of the given year.
    
    Args:
        prediction_dates: list of datetime.date objects
        year: int, target year
    
    Returns:
        int or None: index into prediction_dates, or None if not found
    """
    from datetime import date as dt_date
    target = dt_date(year, TARGET_MONTH, TARGET_DAY)
    
    for i, d in enumerate(prediction_dates):
        if d == target:
            return i
    
    # If exact date not found, find closest December date in that year
    dec_indices = [i for i, d in enumerate(prediction_dates) 
                   if d.year == year and d.month == TARGET_MONTH]
    if dec_indices:
        # Find closest to day 29
        closest = min(dec_indices, key=lambda i: abs(prediction_dates[i].day - TARGET_DAY))
        print(f"  NOTE: Exact Dec 29 {year} not found. Using {prediction_dates[closest]} instead.")
        return closest
    
    return None


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("DECEMBER 29th NORMALIZATION STUDY")
    print("Spatial Temperature Distribution Across Climate Epochs")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load prediction dates
    print("\n[1/4] Loading temporal metadata...")
    df_miroc = pd.read_csv(MIROC_CSV, usecols=['Date'])
    df_miroc['Date'] = pd.to_datetime(df_miroc['Date']).dt.date
    miroc_dates = sorted(df_miroc['Date'].unique())
    
    SEQ_LENGTH = 14
    prediction_dates = miroc_dates[SEQ_LENGTH - 1:]
    
    # Step 2: Load predictions
    print("[2/4] Loading future projection cube...")
    predictions = np.load(NPY_PATH)
    print(f"  Shape: {predictions.shape}")
    
    # Align lengths
    n_preds = min(len(predictions), len(prediction_dates))
    predictions = predictions[:n_preds]
    prediction_dates = prediction_dates[:n_preds]
    
    # Step 3: Extract Dec 29 grids for each target year
    print(f"\n[3/4] Extracting December 29th grids for years: {TARGET_YEARS}")
    
    dec29_grids = {}      # {year: raw 17x17 temperature grid}
    dec29_normalized = {}  # {year: normalized 17x17 grid}
    dec29_tmin = {}        # {year: T_min value used for normalization}
    
    for year in TARGET_YEARS:
        idx = find_dec29_index(prediction_dates, year)
        
        if idx is None:
            print(f"  WARNING: No December data found for {year}. Skipping.")
            continue
        
        # Extract temperature channel (channel index 0 = T_avg)
        temp_grid = predictions[idx, 0, :, :]  # (17, 17) daily max temperature
        
        # Find T_min across the entire spatial grid
        t_min = float(np.nanmin(temp_grid))
        
        # Normalize: T_normalized = T - T_min
        normalized = temp_grid - t_min
        
        dec29_grids[year] = temp_grid
        dec29_normalized[year] = normalized
        dec29_tmin[year] = t_min
        
        print(f"  {year}: T_min = {t_min:.2f} C, T_max = {np.nanmax(temp_grid):.2f} C, "
              f"Range after normalization: [0.00, {np.nanmax(normalized):.2f}] C")
    
    if len(dec29_grids) == 0:
        print("ERROR: No valid December 29 data found for any target year.")
        exit(1)
    
    # Step 4: Plot side-by-side normalized distributions
    print(f"\n[4/4] Generating normalized spatial heatmaps...")
    
    valid_years = [y for y in TARGET_YEARS if y in dec29_normalized]
    n_years = len(valid_years)
    
    # ---- Figure 1: Normalized Heatmaps (consistent color scale) ----
    # Find global max for consistent color scaling across all panels
    global_norm_max = max(np.nanmax(dec29_normalized[y]) for y in valid_years)
    
    fig, axes = plt.subplots(1, n_years, figsize=(7 * n_years, 6))
    if n_years == 1:
        axes = [axes]
    
    fig.suptitle('December 29th: Normalized Temperature Distribution (T - T_min)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    for ax_idx, year in enumerate(valid_years):
        ax = axes[ax_idx]
        grid = dec29_normalized[year]
        t_min = dec29_tmin[year]
        
        # Plot heatmap with consistent color scale
        im = ax.imshow(grid, cmap='YlOrRd', origin='lower',
                       vmin=0, vmax=global_norm_max, aspect='equal')
        
        # Title with normalization info
        ax.set_title(f'{year}\n'
                     f'T_min = {t_min:.1f} C (subtracted)\n'
                     f'Range: [0, {np.nanmax(grid):.1f}] C',
                     fontsize=12, fontweight='bold')
        
        apply_geo_axes(ax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('T - T_min (C)', fontsize=10)
        
        # Annotate max warming pixel
        max_pos = np.unravel_index(np.nanargmax(grid), grid.shape)
        ax.plot(max_pos[1], max_pos[0], 'k*', markersize=12)
        ax.annotate(f'+{grid[max_pos]:.1f} C',
                    xy=(max_pos[1], max_pos[0]),
                    xytext=(max_pos[1] + 1.5, max_pos[0] + 1.5),
                    fontsize=9, fontweight='bold', color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add explanatory annotation at the bottom
    fig.text(0.5, -0.06,
             'Normalization removes the regional baseline warming signal (T_min per grid).\n'
             'The remaining pattern shows relative urban heat island intensification across spatial pixels.\n'
             'Black star (*) marks the pixel with the maximum relative warming.',
             ha='center', fontsize=10, style='italic', color='#555555',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dec29_normalized_heatmaps.png'),
                dpi=150, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"  Saved: dec29_normalized_heatmaps.png")
    
    # ---- Figure 2: Raw vs Normalized Comparison ----
    fig, axes = plt.subplots(2, n_years, figsize=(7 * n_years, 12))
    if n_years == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('December 29th: Raw vs Normalized Temperature Grids',
                 fontsize=16, fontweight='bold')
    
    # Row 1: Raw temperatures
    raw_vmin = min(np.nanmin(dec29_grids[y]) for y in valid_years)
    raw_vmax = max(np.nanmax(dec29_grids[y]) for y in valid_years)
    
    for ax_idx, year in enumerate(valid_years):
        ax = axes[0, ax_idx]
        im = ax.imshow(dec29_grids[year], cmap='RdYlBu_r', origin='lower',
                       vmin=raw_vmin, vmax=raw_vmax, aspect='equal')
        ax.set_title(f'{year} - Raw Temperature', fontsize=12, fontweight='bold')
        apply_geo_axes(ax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (C)', fontsize=9)
    
    # Row 2: Normalized temperatures
    for ax_idx, year in enumerate(valid_years):
        ax = axes[1, ax_idx]
        im = ax.imshow(dec29_normalized[year], cmap='YlOrRd', origin='lower',
                       vmin=0, vmax=global_norm_max, aspect='equal')
        ax.set_title(f'{year} - Normalized (T - T_min)', fontsize=12, fontweight='bold')
        apply_geo_axes(ax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('T - T_min (C)', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dec29_raw_vs_normalized.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dec29_raw_vs_normalized.png")
    
    # ---- Summary Statistics Table ----
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: December 29th Normalization Study")
    print(f"{'=' * 60}")
    print(f"  {'Year':<8} {'T_min (C)':<12} {'T_max (C)':<12} {'Range (C)':<12} {'Max UHI delta':<15}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*15}")
    for year in valid_years:
        raw = dec29_grids[year]
        norm = dec29_normalized[year]
        print(f"  {year:<8} {np.nanmin(raw):<12.2f} {np.nanmax(raw):<12.2f} "
              f"{np.nanmax(raw) - np.nanmin(raw):<12.2f} {np.nanmax(norm):<15.2f}")
    
    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print(f"  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {f}")
    print(f"{'=' * 60}")
