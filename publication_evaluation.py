import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("======================================================")
print("PUBLICATION-GRADE EVALUATION: DOWNSCALER vs ERA5 BASELINE")
print("======================================================")

os.makedirs('evaluation/publication', exist_ok=True)

# ─── 1. LOAD AND ALIGN DATES ───────────────────────────────────────────────
print("\n[1/6] Loading Dates and Aligning Temporal Sequences...")

era5_csv = 'UAE_ERA5_Spatial_Baseline_2015_2025.csv'
df_era5 = pd.read_csv(era5_csv)
df_era5['Date'] = pd.to_datetime(df_era5['Date']).dt.date
era5_dates = sorted(df_era5['Date'].unique())

miroc_csv = 'MIROC6_UAE_SSP245_2015_2025.csv'
df_miroc = pd.read_csv(miroc_csv, usecols=['Date'])
df_miroc['Date'] = pd.to_datetime(df_miroc['Date']).dt.date
miroc_dates = sorted(df_miroc['Date'].unique())

SEQ_LENGTH = 14
prediction_dates = miroc_dates[13:]

valid_dates = sorted(list(set(era5_dates).intersection(set(prediction_dates))))
print(f"  Found {len(valid_dates)} exact matching days between 2015 and 2025.")

era5_indices = [era5_dates.index(d) for d in valid_dates]
miroc_indices = [prediction_dates.index(d) for d in valid_dates]

# ─── 2. LOAD AND SLICE ARRAYS ──────────────────────────────────────────────
print("\n[2/6] Loading Full Arrays and Slicing to Target Window...")

FEATURES = ['T_avg', 'PCP', 'AP', 'RH', 'WS']
UNITS = ['°C', 'mm', 'hPa', '%', 'm/s']

raw_ml_full = np.load('results_future/ssp245/downscaled_ssp245.npy')
fuzzy_ml_full = np.load('results_future/ssp245/uhi_adjusted_full_ssp245.npy')

raw_sub = raw_ml_full[miroc_indices]
fuzzy_sub = fuzzy_ml_full[miroc_indices]

df_era5_sub = df_era5[df_era5['Date'].isin(valid_dates)].sort_values(by=['Date', 'Lat', 'Lon'])
grouped = df_era5_sub.groupby('Date')

era5_data_dict = {}
for date, group in grouped:
    spatial_tensor = []
    for feat in FEATURES:
        mat = group[feat].values.reshape(17, 17)
        spatial_tensor.append(mat)
    era5_data_dict[date] = np.stack(spatial_tensor, axis=0)

era5_sub = np.stack([era5_data_dict[d] for d in valid_dates], axis=0)

# Helper metrics
def mae(pred, truth):
    return float(np.nanmean(np.abs(pred - truth)))

def rmse(pred, truth):
    return float(np.sqrt(np.nanmean((pred - truth)**2)))

def bias(pred, truth):
    return float(np.nanmean(pred - truth))

def r_squared_true(pred, truth):
    p = pred.ravel()
    t = truth.ravel()
    mask = ~(np.isnan(p) | np.isnan(t))
    if mask.sum() < 2: return np.nan
    ss_res = np.sum((t[mask] - p[mask])**2)
    ss_tot = np.sum((t[mask] - np.mean(t[mask]))**2)
    if ss_tot == 0: return np.nan
    return float(1 - ss_res / ss_tot)

def ssim_spatial(pred_stack, truth_stack):
    data_range = float(np.nanmax(truth_stack) - np.nanmin(truth_stack))
    if data_range == 0: return np.nan
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range)**2, (K2 * data_range)**2
    ssim_vals = []
    for t in range(pred_stack.shape[0]):
        p, g = pred_stack[t].astype(float), truth_stack[t].astype(float)
        mu_p, mu_g = np.nanmean(p), np.nanmean(g)
        sig_p, sig_g = np.nanstd(p), np.nanstd(g)
        sig_pg = np.nanmean((p - mu_p)*(g - mu_g))
        num = (2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)
        den = (mu_p**2 + mu_g**2 + C1) * (sig_p**2 + sig_g**2 + C2)
        ssim_vals.append(num / den if den != 0 else 0.0)
    return float(np.mean(ssim_vals))

def compute_metrics(pred, truth, variant):
    results = []
    for ci, (feat, unit) in enumerate(zip(FEATURES, UNITS)):
        p, t = pred[:, ci, :, :], truth[:, ci, :, :]
        results.append({
            'Variant': variant,
            'Variable': feat,
            'Unit': unit,
            'MAE': mae(p, t),
            'RMSE': rmse(p, t),
            'Bias': bias(p, t),
            'R2': r_squared_true(p, t),
            'SSIM': ssim_spatial(p, t)
        })
    return results

# ─── 3. FULL 5-VARIABLE METRICS TABLE ──────────────────────────────────────
print("\n[3/6] Generating Full Metrics Table...")
metrics_raw = compute_metrics(raw_sub, era5_sub, 'Raw CNN-LSTM')
metrics_fuz = compute_metrics(fuzzy_sub, era5_sub, 'Fuzzy-Adjusted')
df_metrics = pd.DataFrame(metrics_raw + metrics_fuz)
df_metrics.to_csv('evaluation/publication/metrics_table.csv', index=False)
print(df_metrics.to_string(index=False))

# ─── 4. TAYLOR DIAGRAM ─────────────────────────────────────────────────────
print("\n[4/6] Generating Taylor Diagram...")
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.set_thetamin(0)
ax.set_thetamax(90)

# Correlation ticks
corr_ticks = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
angles = np.arccos(corr_ticks)
ax.set_xticks(angles)
ax.set_xticklabels([str(c) for c in corr_ticks])
ax.set_title("Taylor Diagram: CNN-LSTM vs ERA5 (2015-2025)", y=1.08)

# CRMSE contours
rs, ts = np.meshgrid(np.linspace(0, 2.5, 100), np.linspace(0, np.pi/2, 100))
crmse = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))
contours = ax.contour(ts, rs, crmse, levels=[0.25, 0.5, 0.75, 1.0, 1.5], colors='gray', linestyles='dashed', alpha=0.5)
ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

# Reference point
ax.plot(0, 1, 'k*', markersize=15, label='ERA5 Reference')

colors = ['red', 'blue', 'green', 'purple', 'orange']
for ci, feat in enumerate(FEATURES):
    e_flat = era5_sub[:, ci, :, :].ravel()
    for variant, p_sub, marker in [('Raw', raw_sub, 'o'), ('Fuzzy', fuzzy_sub, '^')]:
        p_flat = p_sub[:, ci, :, :].ravel()
        std_ratio = np.std(p_flat) / np.std(e_flat)
        corr = np.corrcoef(e_flat, p_flat)[0, 1]
        theta = np.arccos(np.clip(corr, 0, 1))
        ax.plot(theta, std_ratio, marker=marker, color=colors[ci], linestyle='', markersize=10, 
                label=f'{feat} ({variant})' if ci == 0 else f'_{feat} ({variant})')

# Add legend elements manually for cleanliness
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='*', color='k', linestyle='None', markersize=10, label='ERA5 Reference')]
for ci, feat in enumerate(FEATURES):
    legend_elements.append(Line2D([0], [0], marker='s', color=colors[ci], linestyle='None', markersize=8, label=feat))
legend_elements.append(Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='Raw'))
legend_elements.append(Line2D([0], [0], marker='^', color='gray', linestyle='None', markersize=8, label='Fuzzy'))

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1.05))
plt.savefig('evaluation/publication/taylor_diagram.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 5. PER-PIXEL R2 MAPS ──────────────────────────────────────────────────
print("\n[5/6] Generating Spatial R² Maps...")
_LAT_TICKS = [0, 4, 8, 12, 16]
_LAT_LABELS = ['22.0°N', '23.0°N', '24.0°N', '25.0°N', '26.0°N']
_LON_TICKS = [0, 4, 8, 12, 16]
_LON_LABELS = ['52.0°E', '53.0°E', '54.0°E', '55.0°E', '56.0°E']

def apply_geo_axes(ax):
    ax.set_yticks(_LAT_TICKS)
    ax.set_yticklabels(_LAT_LABELS, fontsize=8)
    ax.set_xticks(_LON_TICKS)
    ax.set_xticklabels(_LON_LABELS, fontsize=8)

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle('Spatial R² (True Coefficient of Determination) vs ERA5 (2015-2025)', fontsize=16)

for ci, feat in enumerate(FEATURES):
    for ri, (variant, p_sub) in enumerate([('Raw CNN-LSTM', raw_sub), ('Fuzzy-Adjusted', fuzzy_sub)]):
        r2_map = np.zeros((17, 17))
        for i in range(17):
            for j in range(17):
                e_ts = era5_sub[:, ci, i, j]
                p_ts = p_sub[:, ci, i, j]
                ss_res = np.sum((e_ts - p_ts)**2)
                ss_tot = np.sum((e_ts - np.mean(e_ts))**2)
                r2_map[i, j] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        ax = axes[ri, ci]
        im = ax.imshow(r2_map, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
        ax.set_title(f"{feat} ({variant})")
        apply_geo_axes(ax)
        if ci == 4:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('evaluation/publication/spatial_r2_maps.png', dpi=150)
plt.close()

# ─── 6. ALL-VARIABLE SEASONALITY & EXTREMES ────────────────────────────────
print("\n[6/6] Generating Seasonality and Extreme Percentiles...")
months = np.array([d.month for d in valid_dates])

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('Monthly Seasonality (UAE Spatial Mean, 2015-2025)', fontsize=16)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for ci, (feat, unit) in enumerate(zip(FEATURES, UNITS)):
    monthly_era5, monthly_raw, monthly_fuz = [], [], []
    for m in range(1, 13):
        mask = (months == m)
        monthly_era5.append(era5_sub[mask, ci].mean())
        monthly_raw.append(raw_sub[mask, ci].mean())
        monthly_fuz.append(fuzzy_sub[mask, ci].mean())
    
    ax = axes[ci]
    ax.plot(month_labels, monthly_era5, 'k-o', label='ERA5')
    ax.plot(month_labels, monthly_raw, 'b--s', label='Raw')
    ax.plot(month_labels, monthly_fuz, 'r--^', label='Fuzzy')
    ax.set_title(f"{feat} ({unit})")
    ax.grid(True, alpha=0.3)
    if ci == 0: ax.legend()

plt.tight_layout()
plt.savefig('evaluation/publication/all_variable_seasonality.png', dpi=150)
plt.close()

# Extremes
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Extreme Percentiles (P95 and P99) across all pixels and days', fontsize=16)

x = np.arange(len(FEATURES))
width = 0.25

p95_era5 = [np.nanpercentile(era5_sub[:, ci], 95) for ci in range(len(FEATURES))]
p95_raw = [np.nanpercentile(raw_sub[:, ci], 95) for ci in range(len(FEATURES))]
p95_fuz = [np.nanpercentile(fuzzy_sub[:, ci], 95) for ci in range(len(FEATURES))]

p99_era5 = [np.nanpercentile(era5_sub[:, ci], 99) for ci in range(len(FEATURES))]
p99_raw = [np.nanpercentile(raw_sub[:, ci], 99) for ci in range(len(FEATURES))]
p99_fuz = [np.nanpercentile(fuzzy_sub[:, ci], 99) for ci in range(len(FEATURES))]

axes[0].bar(x - width, p95_era5, width, label='ERA5', color='black')
axes[0].bar(x, p95_raw, width, label='Raw CNN', color='steelblue')
axes[0].bar(x + width, p95_fuz, width, label='Fuzzy CNN', color='darkorange')
axes[0].set_xticks(x)
axes[0].set_xticklabels(FEATURES)
axes[0].set_title('95th Percentile')
axes[0].legend()

axes[1].bar(x - width, p99_era5, width, label='ERA5', color='black')
axes[1].bar(x, p99_raw, width, label='Raw CNN', color='steelblue')
axes[1].bar(x + width, p99_fuz, width, label='Fuzzy CNN', color='darkorange')
axes[1].set_xticks(x)
axes[1].set_xticklabels(FEATURES)
axes[1].set_title('99th Percentile')

plt.tight_layout()
plt.savefig('evaluation/publication/extreme_percentiles.png', dpi=150)
plt.close()

print("\n======================================================")
print("  EVALUATION COMPLETE - RESULTS SAVED TO /evaluation/publication/")
print("======================================================")
