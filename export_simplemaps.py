"""
export_simplemaps.py - Convert 17x17 .npy Future Predictions to SimpleMaps mapdata.js
=====================================================================================
Reads the downscaled 17x17 future projection grids (.npy), aggregates them into
per-Emirate averages by mapping pixel coordinates to the 7 UAE Emirates, and
exports a Javascript file compatible with SimpleMaps' countrymap_mapdata format.

Output: mapdata.js (ready for frontend integration)

Emirate Pixel Mapping:
    The 17x17 grid covers UAE lat [22.0, 26.2] x lon [52.0, 56.2].
    Each pixel is assigned to the Emirate whose geographic bounding box it falls within.
    Pixels outside all Emirate bounds are excluded from aggregation.
"""
import numpy as np
import os
import json
from datetime import date

# ─── CONFIG ───────────────────────────────────────────────
FEATURES = ['T_avg', 'PCP', 'AP', 'RH', 'WS']

# Default paths (can be overridden)
NPY_PATH = 'results_future/ssp245/uhi_adjusted_full_ssp245.npy'
MIROC_CSV = 'MIROC6_UAE_SSP245_2015_2100.csv'
OUTPUT_DIR = 'evaluation/simplemaps'

# ─── UAE GRID COORDINATES ─────────────────────────────────
# The 17x17 grid spans these latitude and longitude ranges
LATS = np.linspace(22.0, 26.2, 17)  # South to North (row index 0 = 22.0N)
LONS = np.linspace(52.0, 56.2, 17)  # West to East   (col index 0 = 52.0E)

# ─── EMIRATE GEOGRAPHIC BOUNDING BOXES ────────────────────
# Approximate bounding boxes for each Emirate (lat_min, lat_max, lon_min, lon_max)
# These are simplified rectangles; real borders are irregular but this captures
# the dominant spatial coverage for each Emirate on the 17x17 grid.
EMIRATE_BOUNDS = {
    'Abu Dhabi': {
        'lat_min': 22.0, 'lat_max': 24.5,
        'lon_min': 52.0, 'lon_max': 55.0,
        'simplemaps_id': 'AE-AZ',
        'name': 'Abu Dhabi'
    },
    'Dubai': {
        'lat_min': 24.8, 'lat_max': 25.4,
        'lon_min': 55.0, 'lon_max': 55.6,
        'simplemaps_id': 'AE-DU',
        'name': 'Dubai'
    },
    'Sharjah': {
        'lat_min': 25.0, 'lat_max': 25.6,
        'lon_min': 55.3, 'lon_max': 55.9,
        'simplemaps_id': 'AE-SH',
        'name': 'Sharjah'
    },
    'Ajman': {
        'lat_min': 25.3, 'lat_max': 25.5,
        'lon_min': 55.4, 'lon_max': 55.6,
        'simplemaps_id': 'AE-AJ',
        'name': 'Ajman'
    },
    'Fujairah': {
        'lat_min': 25.0, 'lat_max': 25.5,
        'lon_min': 56.0, 'lon_max': 56.4,
        'simplemaps_id': 'AE-FU',
        'name': 'Fujairah'
    },
    'Ras Al Khaimah': {
        'lat_min': 25.5, 'lat_max': 26.2,
        'lon_min': 55.5, 'lon_max': 56.2,
        'simplemaps_id': 'AE-RK',
        'name': 'Ras Al Khaimah'
    },
    'Umm Al Quwain': {
        'lat_min': 25.4, 'lat_max': 25.7,
        'lon_min': 55.5, 'lon_max': 55.8,
        'simplemaps_id': 'AE-UQ',
        'name': 'Umm Al Quwain'
    }
}


def get_pixel_mask(emirate_bounds):
    """
    Generate a boolean mask (17x17) indicating which pixels fall within the
    geographic bounding box of a given Emirate.
    
    Args:
        emirate_bounds: dict with lat_min, lat_max, lon_min, lon_max
    
    Returns:
        mask: np.ndarray of shape (17, 17), True for pixels inside the Emirate
    """
    mask = np.zeros((17, 17), dtype=bool)
    for i, lat in enumerate(LATS):
        for j, lon in enumerate(LONS):
            if (emirate_bounds['lat_min'] <= lat <= emirate_bounds['lat_max'] and
                emirate_bounds['lon_min'] <= lon <= emirate_bounds['lon_max']):
                mask[i, j] = True
    return mask


def temperature_to_hex_color(temp_c):
    """
    Convert a temperature value to a hex color for heatmap visualization.
    Uses a blue (cold) -> yellow -> red (hot) gradient scaled to UAE temps.
    
    Scale: 20C = cool blue, 35C = yellow, 50C+ = deep red
    """
    # Normalize to 0-1 range (20C to 50C)
    t_norm = np.clip((temp_c - 20.0) / 30.0, 0.0, 1.0)
    
    if t_norm < 0.5:
        # Blue to Yellow transition
        ratio = t_norm * 2.0
        r = int(255 * ratio)
        g = int(255 * ratio)
        b = int(255 * (1.0 - ratio))
    else:
        # Yellow to Red transition
        ratio = (t_norm - 0.5) * 2.0
        r = 255
        g = int(255 * (1.0 - ratio))
        b = 0
    
    return f'#{r:02X}{g:02X}{b:02X}'


def aggregate_emirate_temperatures(predictions, target_year, prediction_dates):
    """
    Compute the annual mean temperature for each Emirate for a given year.
    
    Args:
        predictions: np.ndarray of shape (T, 5, 17, 17)
        target_year: int, the year to extract
        prediction_dates: list of date objects corresponding to predictions
    
    Returns:
        dict: {emirate_name: mean_temperature}
    """
    # Find indices for the target year
    year_mask = [i for i, d in enumerate(prediction_dates) if d.year == target_year]
    
    if len(year_mask) == 0:
        print(f"  WARNING: No data found for year {target_year}")
        return {}
    
    # Extract temperature channel (index 0 = T_avg) for the target year
    temp_grids = predictions[year_mask, 0, :, :]  # (N_days, 17, 17)
    annual_mean_grid = np.mean(temp_grids, axis=0)  # (17, 17)
    
    emirate_temps = {}
    for emirate_name, bounds in EMIRATE_BOUNDS.items():
        mask = get_pixel_mask(bounds)
        n_pixels = mask.sum()
        
        if n_pixels == 0:
            # Emirate too small for the grid resolution; use nearest pixel
            center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2.0
            center_lon = (bounds['lon_min'] + bounds['lon_max']) / 2.0
            lat_idx = np.argmin(np.abs(LATS - center_lat))
            lon_idx = np.argmin(np.abs(LONS - center_lon))
            emirate_temps[emirate_name] = float(annual_mean_grid[lat_idx, lon_idx])
        else:
            emirate_temps[emirate_name] = float(annual_mean_grid[mask].mean())
    
    return emirate_temps


def generate_mapdata_js(emirate_temps, target_year, output_path):
    """
    Generate a SimpleMaps-compatible mapdata.js file with heatmap colors
    and hover tooltips showing predicted temperatures per Emirate.
    
    Args:
        emirate_temps: dict {emirate_name: mean_temperature}
        target_year: int
        output_path: str, path to write the .js file
    """
    # Build the state_specific section
    state_specific = {}
    for emirate_name, bounds in EMIRATE_BOUNDS.items():
        temp = emirate_temps.get(emirate_name, 0.0)
        color = temperature_to_hex_color(temp)
        
        state_specific[bounds['simplemaps_id']] = {
            'name': bounds['name'],
            'color': color,
            'hover_color': color,
            'url': '',
            'description': f'Projected Avg Temp ({target_year}): {temp:.1f} C'
        }
    
    # Build the full mapdata object
    mapdata = {
        'main_settings': {
            'type': 'country',
            'code': 'AE',
            'width': 'responsive',
            'background_color': '#FFFFFF',
            'background_transparent': 'yes',
            'border_color': '#333333',
            'state_description': f'CNN-LSTM + Fuzzy Physics Projection ({target_year})',
            'state_color': '#88A4BC',
            'state_hover_color': '#3B729F',
            'state_url': '',
            'all_states_inactive': 'no',
            'location_description': 'Location',
            'location_color': '#FF4444',
            'location_opacity': '0.8',
            'location_url': '',
            'location_size': '20',
            'location_type': 'circle',
            'location_border_color': '#FFFFFF',
            'all_locations_inactive': 'no',
            'url_new_tab': 'yes',
            'auto_load': 'yes',
            'popups': 'detect',
            'state_image_url': '',
            'state_image_position': '',
            'location_image_url': '',
            'zoom': 'yes',
            'initial_zoom': '-1',
            'initial_zoom_solo': 'no',
            'label_color': '#d5ddec',
            'label_hover_color': '#d5ddec',
            'label_size': '22',
            'label_font': 'Arial',
            'hide_labels': 'no',
            'hide_eastern_labels': 'no',
            'manual_zoom': 'no',
            'back_image': 'no',
            'arrow_color': '#cecece',
            'arrow_color_border': '#808080',
            'initial_back': 'no',
            'initial_zoom_level': '-1',
            'js_hooks_enabled': 'no'
        },
        'state_specific': state_specific,
        'locations': {},
        'labels': {}
    }
    
    # Write as JavaScript variable assignment
    js_content = f"// Auto-generated by export_simplemaps.py\n"
    js_content += f"// CNN-LSTM + Fuzzy Physics Climate Projection for UAE ({target_year})\n"
    js_content += f"// Generated on: {date.today().isoformat()}\n\n"
    js_content += f"var simplemaps_countrymap_mapdata = {json.dumps(mapdata, indent=2)};\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"  Saved: {output_path}")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == '__main__':
    import pandas as pd
    
    print("=" * 60)
    print("SIMPLEMAPS EXPORTER: 17x17 Predictions -> mapdata.js")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load prediction dates from MIROC6 CSV
    print("\n[1/3] Loading prediction dates...")
    df_miroc = pd.read_csv(MIROC_CSV, usecols=['Date'])
    df_miroc['Date'] = pd.to_datetime(df_miroc['Date']).dt.date
    miroc_dates = sorted(df_miroc['Date'].unique())
    
    SEQ_LENGTH = 14
    prediction_dates = miroc_dates[SEQ_LENGTH - 1:]
    
    # Load predictions
    print("[2/3] Loading .npy predictions...")
    predictions = np.load(NPY_PATH)
    print(f"  Shape: {predictions.shape}")
    print(f"  Date range: {prediction_dates[0]} -> {prediction_dates[-1]}")
    
    # Ensure alignment
    n_preds = min(len(predictions), len(prediction_dates))
    predictions = predictions[:n_preds]
    prediction_dates = prediction_dates[:n_preds]
    
    # Generate mapdata.js for key projection years
    print("\n[3/3] Generating mapdata.js files...")
    target_years = [2030, 2050, 2070, 2090]
    
    for year in target_years:
        print(f"\n  --- Year {year} ---")
        emirate_temps = aggregate_emirate_temperatures(predictions, year, prediction_dates)
        
        if not emirate_temps:
            print(f"  Skipping {year} (no data)")
            continue
        
        # Print summary
        for emirate, temp in emirate_temps.items():
            color = temperature_to_hex_color(temp)
            print(f"    {emirate:20s}: {temp:.1f} C  [{color}]")
        
        output_path = os.path.join(OUTPUT_DIR, f'mapdata_{year}.js')
        generate_mapdata_js(emirate_temps, year, output_path)
    
    print(f"\n{'=' * 60}")
    print(f"EXPORT COMPLETE - Files saved to {OUTPUT_DIR}/")
    print(f"{'=' * 60}")
