import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')

print("======================================================")
print("BASELINE COMPARISON: CNN-LSTM vs CLASSICAL INTERPOLATION")
print("======================================================")

os.makedirs('evaluation/baselines', exist_ok=True)

# ─── 1. LOAD AND ALIGN DATES ───────────────────────────────────────────────
print("\n[1/5] Loading data and aligning sequences...")

era5_csv = 'UAE_ERA5_Spatial_Baseline_2015_2025.csv'
df_era5 = pd.read_csv(era5_csv)
df_era5['Date'] = pd.to_datetime(df_era5['Date']).dt.date
era5_dates = sorted(df_era5['Date'].unique())

miroc_csv = 'MIROC6_UAE_SSP245_2015_2025.csv'
df_miroc = pd.read_csv(miroc_csv)
df_miroc['Date'] = pd.to_datetime(df_miroc['Date']).dt.date
miroc_dates = sorted(df_miroc['Date'].unique())

SEQ_LENGTH = 14
prediction_dates = miroc_dates[13:]

valid_dates = sorted(list(set(era5_dates).intersection(set(prediction_dates))))
print(f"  Found {len(valid_dates)} exact matching days between 2015 and 2025.")

era5_indices = [era5_dates.index(d) for d in valid_dates]
miroc_indices = [prediction_dates.index(d) for d in valid_dates]

FEATURES = ['T_avg', 'PCP', 'AP', 'RH', 'WS']
UNITS = ['°C', 'mm', 'hPa', '%', 'm/s']

raw_ml_full = np.load('results_future/ssp245/downscaled_ssp245.npy')
fuzzy_ml_full = np.load('results_future/ssp245/uhi_adjusted_full_ssp245.npy')

raw_sub = raw_ml_full[miroc_indices]
fuzzy_sub = fuzzy_ml_full[miroc_indices]

# ERA5 17x17
df_era5_sub = df_era5[df_era5['Date'].isin(valid_dates)].sort_values(by=['Date', 'Lat', 'Lon'])
grouped_era5 = df_era5_sub.groupby('Date')

era5_data_dict = {}
for date, group in grouped_era5:
    spatial_tensor = []
    for feat in FEATURES:
        mat = group[feat].values.reshape(17, 17)
        spatial_tensor.append(mat)
    era5_data_dict[date] = np.stack(spatial_tensor, axis=0)
era5_sub = np.stack([era5_data_dict[d] for d in valid_dates], axis=0)

# MIROC6 3x3 for valid dates
df_miroc_sub = df_miroc[df_miroc['Date'].isin(valid_dates)].sort_values(by=['Date', 'Lat', 'Lon'])
grouped_miroc = df_miroc_sub.groupby('Date')

miroc_data_dict = {}
for date, group in grouped_miroc:
    spatial_tensor = []
    for feat in FEATURES:
        mat = group[feat].values.reshape(3, 3)
        spatial_tensor.append(mat)
    miroc_data_dict[date] = np.stack(spatial_tensor, axis=0)

print("\n[2/5] Generating classical baselines (Bilinear, Bicubic)...")
def upscale_grid(grid_CxHxW, target_size=17, order=1):
    C, H, W = grid_CxHxW.shape
    result = np.zeros((C, target_size, target_size))
    for c in range(C):
        zoomed = zoom(grid_CxHxW[c], (target_size / H, target_size / W), order=order)
        result[c] = zoomed[:target_size, :target_size]
    return result

bilinear_sub = np.zeros_like(era5_sub)
bicubic_sub = np.zeros_like(era5_sub)

for i, d in enumerate(valid_dates):
    gcm_3x3 = miroc_data_dict[d]
    bilinear_sub[i] = upscale_grid(gcm_3x3, target_size=17, order=1)
    bicubic_sub[i] = upscale_grid(gcm_3x3, target_size=17, order=3)

# ─── 3. METRICS ───────────────────────────────────────────────────────────
print("\n[3/5] Computing metrics...")
def mae(pred, truth): return float(np.nanmean(np.abs(pred - truth)))
def rmse(pred, truth): return float(np.sqrt(np.nanmean((pred - truth)**2)))
def bias(pred, truth): return float(np.nanmean(pred - truth))
def r_squared_true(pred, truth):
    p = pred.ravel()
    t = truth.ravel()
    mask = ~(np.isnan(p) | np.isnan(t))
    if mask.sum() < 2: return np.nan
    ss_res = np.sum((t[mask] - p[mask])**2)
    ss_tot = np.sum((t[mask] - np.mean(t[mask]))**2)
    if ss_tot == 0: return np.nan
    return float(1 - ss_res / ss_tot)

def get_metrics_for_stack(pred_stack, method_name):
    res = []
    for ci, (feat, unit) in enumerate(zip(FEATURES, UNITS)):
        p, t = pred_stack[:, ci], era5_sub[:, ci]
        res.append({
            'Method': method_name, 'Variable': feat, 'Unit': unit,
            'MAE': mae(p, t), 'RMSE': rmse(p, t),
            'R2': r_squared_true(p, t), 'Bias': bias(p, t)
        })
    return res

m_bilinear = get_metrics_for_stack(bilinear_sub, 'Bilinear')
m_bicubic = get_metrics_for_stack(bicubic_sub, 'Bicubic')
m_raw = get_metrics_for_stack(raw_sub, 'CNN-LSTM (Raw)')
m_fuz = get_metrics_for_stack(fuzzy_sub, 'CNN-LSTM + Fuzzy')

all_metrics = m_bilinear + m_bicubic + m_raw + m_fuz
df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_csv('evaluation/baselines/baseline_metrics_table.csv', index=False)
print(df_metrics.to_string(index=False))

# ─── 4. BAR CHART ──────────────────────────────────────────────────────────
print("\n[4/5] Plotting grouped bar chart...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Baseline Comparison (2015-2025 Validation)', fontsize=16)

methods = ['Bilinear', 'Bicubic', 'CNN-LSTM (Raw)', 'CNN-LSTM + Fuzzy']
colors = ['#95A5A6', '#5DADE2', '#3498DB', '#E67E22']
metrics_to_plot = ['MAE', 'RMSE', 'R2', 'Bias']

x = np.arange(len(FEATURES))
width = 0.2

for mi, metric in enumerate(metrics_to_plot):
    ax = axes.flatten()[mi]
    for i, method in enumerate(methods):
        method_data = df_metrics[df_metrics['Method'] == method]
        vals = [method_data[method_data['Variable'] == f][metric].values[0] for f in FEATURES]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=method, color=colors[i])
    
    ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES)
    if mi == 0: ax.legend()

plt.tight_layout()
plt.savefig('evaluation/baselines/baseline_comparison_bars.png', dpi=150)
plt.close()

# ─── 5. SPATIAL ERROR MAPS ───────────────────────────────────────────────
print("\n[5/5] Generating spatial error maps for Temperature...")
_LAT_TICKS = [0, 4, 8, 12, 16]
_LAT_LABELS = ['22.0°N', '23.0°N', '24.0°N', '25.0°N', '26.0°N']
_LON_TICKS = [0, 4, 8, 12, 16]
_LON_LABELS = ['52.0°E', '53.0°E', '54.0°E', '55.0°E', '56.0°E']

def apply_geo_axes(ax):
    ax.set_yticks(_LAT_TICKS)
    ax.set_yticklabels(_LAT_LABELS, fontsize=8)
    ax.set_xticks(_LON_TICKS)
    ax.set_xticklabels(_LON_LABELS, fontsize=8)

idx_t = FEATURES.index('T_avg')
err_bilinear = np.mean(np.abs(bilinear_sub[:, idx_t] - era5_sub[:, idx_t]), axis=0)
err_bicubic = np.mean(np.abs(bicubic_sub[:, idx_t] - era5_sub[:, idx_t]), axis=0)
err_raw = np.mean(np.abs(raw_sub[:, idx_t] - era5_sub[:, idx_t]), axis=0)
err_fuz = np.mean(np.abs(fuzzy_sub[:, idx_t] - era5_sub[:, idx_t]), axis=0)

vmax = max(err_bilinear.max(), err_bicubic.max(), err_raw.max(), err_fuz.max())

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Spatial MAE (Temperature) vs ERA5 (2015-2025)', fontsize=14)

for ax, err_map, title in zip(
    axes, 
    [err_bilinear, err_bicubic, err_raw, err_fuz], 
    ['Bilinear Interpolation', 'Bicubic Interpolation', 'CNN-LSTM (Raw)', 'CNN-LSTM + Fuzzy']
):
    im = ax.imshow(err_map, cmap='Reds', origin='lower', vmin=0, vmax=vmax)
    ax.set_title(f"{title}\nMean MAE: {err_map.mean():.2f}°C")
    apply_geo_axes(ax)

plt.colorbar(im, ax=axes, fraction=0.015, pad=0.04, location='right')
plt.savefig('evaluation/baselines/baseline_spatial_errors.png', dpi=150)
plt.close()

# ─── IMPROVEMENT PERCENTAGES ──────────────────────────────────────────────
imp_rows = []
for feat in FEATURES:
    bl = df_metrics[(df_metrics['Variable'] == feat) & (df_metrics['Method'] == 'Bilinear')]
    bc = df_metrics[(df_metrics['Variable'] == feat) & (df_metrics['Method'] == 'Bicubic')]
    fuz = df_metrics[(df_metrics['Variable'] == feat) & (df_metrics['Method'] == 'CNN-LSTM + Fuzzy')]
    
    bl_mae, bc_mae, fuz_mae = bl['MAE'].values[0], bc['MAE'].values[0], fuz['MAE'].values[0]
    bl_r2, bc_r2, fuz_r2 = bl['R2'].values[0], bc['R2'].values[0], fuz['R2'].values[0]
    
    imp_mae_bl = (bl_mae - fuz_mae) / bl_mae * 100
    imp_mae_bc = (bc_mae - fuz_mae) / bc_mae * 100
    imp_r2_bl = (fuz_r2 - bl_r2) / (1 - bl_r2 + 1e-9) * 100
    imp_r2_bc = (fuz_r2 - bc_r2) / (1 - bc_r2 + 1e-9) * 100
    
    imp_rows.append({
        'Variable': feat,
        'Imp_MAE_vs_Bilinear(%)': imp_mae_bl,
        'Imp_MAE_vs_Bicubic(%)': imp_mae_bc,
        'Imp_R2_vs_Bilinear(%)': imp_r2_bl,
        'Imp_R2_vs_Bicubic(%)': imp_r2_bc
    })

df_imp = pd.DataFrame(imp_rows)
df_imp.to_csv('evaluation/baselines/baseline_improvement.csv', index=False)
print("\nImprovement over Baselines:")
print(df_imp.to_string(index=False))

print("\n======================================================")
print("  EVALUATION COMPLETE - RESULTS SAVED TO /evaluation/baselines/")
print("======================================================")
