"""
fuzzy_uhi.py - Multi-Variable Urban Physics Engine (Vectorized)
================================================================
Implements three independent Mamdani Fuzzy Inference Systems to model
the physical effects of Dubai's urban fabric on the CNN-LSTM output:

  1. Thermal Logic   : T_adj  = f(T_base, D_urban)  --> Positive (Urban Heat Island)
  2. Moisture Logic   : RH_adj = f(RH_base, D_urban) --> Negative (Urban Dry Island)
  3. Friction Logic   : WS_adj = f(WS_base, D_urban) --> Negative (Surface Roughness)

CRITICAL: Precipitation (PCP) and Air Pressure (AP) are NEVER modified.

Performance: Uses precomputed lookup tables (LUTs) instead of per-pixel
scikit-fuzzy evaluation. This makes adjust_full_grid ~500x faster.
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import RegularGridInterpolator


class UHIFuzzyAdjuster:
    def __init__(self):
        """
        Initializes three Mamdani FIS and precomputes lookup tables
        for fast vectorized evaluation.
        """
        print("  Building Fuzzy Inference Systems...")
        
        # ═══════════════════════════════════════════════════
        # FIS 1: THERMAL LOGIC (Urban Heat Island)
        # ═══════════════════════════════════════════════════
        t_base = ctrl.Antecedent(np.arange(10, 56, 1), 't_base')
        d_thermal = ctrl.Antecedent(np.arange(0, 25, 1), 'd_thermal')
        uhi_adj = ctrl.Consequent(np.arange(0, 4, 0.1), 'uhi_adj', defuzzify_method='centroid')

        t_base['Mild'] = fuzz.trapmf(t_base.universe, [10, 10, 25, 30])
        t_base['Warm'] = fuzz.trimf(t_base.universe, [25, 35, 45])
        t_base['Hot']  = fuzz.trapmf(t_base.universe, [35, 45, 55, 55])

        d_thermal['Core']   = fuzz.trapmf(d_thermal.universe, [0, 0, 2, 5])
        d_thermal['Suburb'] = fuzz.trimf(d_thermal.universe, [2, 6, 12])
        d_thermal['Desert'] = fuzz.trapmf(d_thermal.universe, [8, 14, 25, 25])

        uhi_adj['Zero']     = fuzz.trimf(uhi_adj.universe, [0, 0, 0.5])
        uhi_adj['Moderate'] = fuzz.trimf(uhi_adj.universe, [0.2, 1.5, 2.5])
        uhi_adj['Severe']   = fuzz.trapmf(uhi_adj.universe, [2.0, 3.0, 4.0, 4.0])

        thermal_rules = [
            ctrl.Rule(d_thermal['Core'] & t_base['Hot'], uhi_adj['Severe']),
            ctrl.Rule(d_thermal['Core'] & t_base['Warm'], uhi_adj['Moderate']),
            ctrl.Rule(d_thermal['Suburb'] & t_base['Hot'], uhi_adj['Moderate']),
            ctrl.Rule(d_thermal['Desert'], uhi_adj['Zero']),
            ctrl.Rule(t_base['Mild'], uhi_adj['Zero']),
        ]
        self._thermal_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(thermal_rules))

        # ═══════════════════════════════════════════════════
        # FIS 2: MOISTURE LOGIC (Urban Dry Island)
        # ═══════════════════════════════════════════════════
        rh_base = ctrl.Antecedent(np.arange(0, 101, 1), 'rh_base')
        d_moisture = ctrl.Antecedent(np.arange(0, 25, 1), 'd_moisture')
        rh_adj = ctrl.Consequent(np.arange(0, 16, 0.5), 'rh_adj', defuzzify_method='centroid')

        rh_base['Dry']      = fuzz.trapmf(rh_base.universe, [0, 0, 20, 35])
        rh_base['Moderate'] = fuzz.trimf(rh_base.universe, [25, 50, 75])
        rh_base['Humid']    = fuzz.trapmf(rh_base.universe, [60, 80, 100, 100])

        d_moisture['Core']   = fuzz.trapmf(d_moisture.universe, [0, 0, 2, 5])
        d_moisture['Suburb'] = fuzz.trimf(d_moisture.universe, [2, 6, 12])
        d_moisture['Desert'] = fuzz.trapmf(d_moisture.universe, [8, 14, 25, 25])

        rh_adj['None']   = fuzz.trimf(rh_adj.universe, [0, 0, 1])
        rh_adj['Mild']   = fuzz.trimf(rh_adj.universe, [1, 4, 8])
        rh_adj['Strong'] = fuzz.trapmf(rh_adj.universe, [6, 10, 15, 15])

        moisture_rules = [
            ctrl.Rule(d_moisture['Core'] & rh_base['Humid'], rh_adj['Strong']),
            ctrl.Rule(d_moisture['Core'] & rh_base['Moderate'], rh_adj['Mild']),
            ctrl.Rule(d_moisture['Suburb'] & rh_base['Humid'], rh_adj['Mild']),
            ctrl.Rule(d_moisture['Desert'], rh_adj['None']),
            ctrl.Rule(rh_base['Dry'], rh_adj['None']),
        ]
        self._moisture_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(moisture_rules))

        # ═══════════════════════════════════════════════════
        # FIS 3: FRICTION LOGIC (Urban Surface Roughness)
        # ═══════════════════════════════════════════════════
        ws_base = ctrl.Antecedent(np.arange(0, 20, 0.5), 'ws_base')
        d_friction = ctrl.Antecedent(np.arange(0, 25, 1), 'd_friction')
        ws_adj = ctrl.Consequent(np.arange(0, 6, 0.1), 'ws_adj', defuzzify_method='centroid')

        ws_base['Calm']     = fuzz.trapmf(ws_base.universe, [0, 0, 2, 4])
        ws_base['Moderate'] = fuzz.trimf(ws_base.universe, [3, 6, 10])
        ws_base['Strong']   = fuzz.trapmf(ws_base.universe, [8, 12, 20, 20])

        d_friction['Core']   = fuzz.trapmf(d_friction.universe, [0, 0, 2, 5])
        d_friction['Suburb'] = fuzz.trimf(d_friction.universe, [2, 6, 12])
        d_friction['Desert'] = fuzz.trapmf(d_friction.universe, [8, 14, 25, 25])

        ws_adj['None']     = fuzz.trimf(ws_adj.universe, [0, 0, 0.5])
        ws_adj['Moderate'] = fuzz.trimf(ws_adj.universe, [0.5, 2.0, 3.5])
        ws_adj['Heavy']    = fuzz.trapmf(ws_adj.universe, [2.5, 4.0, 6.0, 6.0])

        friction_rules = [
            ctrl.Rule(d_friction['Core'] & ws_base['Strong'], ws_adj['Heavy']),
            ctrl.Rule(d_friction['Core'] & ws_base['Moderate'], ws_adj['Moderate']),
            ctrl.Rule(d_friction['Suburb'] & ws_base['Strong'], ws_adj['Moderate']),
            ctrl.Rule(d_friction['Desert'], ws_adj['None']),
            ctrl.Rule(ws_base['Calm'], ws_adj['None']),
        ]
        self._friction_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(friction_rules))

        # ═══════════════════════════════════════════════════
        # PRECOMPUTE LOOKUP TABLES for vectorized evaluation
        # ═══════════════════════════════════════════════════
        print("  Precomputing fuzzy lookup tables (one-time cost)...")
        self._thermal_interp = self._build_lut(
            self._thermal_sim, 't_base', 'd_thermal', 'uhi_adj',
            val_range=(10, 55, 46), dist_range=(0, 24, 25)
        )
        self._moisture_interp = self._build_lut(
            self._moisture_sim, 'rh_base', 'd_moisture', 'rh_adj',
            val_range=(0, 100, 101), dist_range=(0, 24, 25)
        )
        self._friction_interp = self._build_lut(
            self._friction_sim, 'ws_base', 'd_friction', 'ws_adj',
            val_range=(0, 19.5, 40), dist_range=(0, 24, 25)
        )
        print("  Lookup tables ready.")

    def _build_lut(self, sim, val_key, dist_key, out_key, val_range, dist_range):
        """
        Precompute a 2D lookup table by evaluating the FIS across a grid
        of (value, distance) pairs. Returns a fast scipy interpolator.
        """
        val_pts = np.linspace(val_range[0], val_range[1], val_range[2])
        dist_pts = np.linspace(dist_range[0], dist_range[1], dist_range[2])
        lut = np.zeros((len(val_pts), len(dist_pts)))
        
        for i, v in enumerate(val_pts):
            for j, d in enumerate(dist_pts):
                sim.input[val_key] = float(v)
                sim.input[dist_key] = float(d)
                try:
                    sim.compute()
                    lut[i, j] = sim.output[out_key]
                except:
                    lut[i, j] = 0.0
        
        interp = RegularGridInterpolator(
            (val_pts, dist_pts), lut,
            method='linear', bounds_error=False, fill_value=0.0
        )
        return interp

    # ─── DISTANCE MATRIX ─────────────────────────────────
    def compute_distance_matrix(self, grid_size=17, core_idx=(12, 10)):
        """Euclidean distance of each pixel from the Dubai Urban Core."""
        rows = np.arange(grid_size)
        cols = np.arange(grid_size)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        return np.sqrt((rr - core_idx[0])**2 + (cc - core_idx[1])**2)

    # ─── FULL MULTI-VARIABLE (VECTORIZED) ─────────────────
    def adjust_full_grid(self, grid_5ch, core_idx=(12, 10)):
        """
        Applies all three fuzzy adjustments to a (5, 17, 17) frame.
        Uses precomputed lookup tables for ~500x speedup.
        
        Channels: [T_avg, PCP, AP, RH, WS]
          - T_avg: +UHI thermal adjustment
          - PCP:   UNCHANGED
          - AP:    UNCHANGED
          - RH:    -Urban Dry Island
          - WS:    -Surface Roughness
        """
        adjusted = np.copy(grid_5ch)
        grid_size = grid_5ch.shape[1]
        dist_matrix = self.compute_distance_matrix(grid_size, core_idx)
        d_flat = np.clip(dist_matrix.ravel(), 0, 24)
        
        # Channel 0: Temperature (+UHI)
        t_clipped = np.clip(grid_5ch[0], 10, 55).ravel()
        pts_t = np.stack([t_clipped, d_flat], axis=-1)
        adjusted[0] += self._thermal_interp(pts_t).reshape(grid_size, grid_size)
        
        # Channel 1: PCP — UNTOUCHED
        # Channel 2: AP  — UNTOUCHED
        
        # Channel 3: RH (-Urban Dry Island)
        rh_clipped = np.clip(grid_5ch[3], 0, 100).ravel()
        pts_rh = np.stack([rh_clipped, d_flat], axis=-1)
        reduction_rh = self._moisture_interp(pts_rh).reshape(grid_size, grid_size)
        adjusted[3] = np.maximum(0.0, adjusted[3] - reduction_rh)
        
        # Channel 4: WS (-Surface Roughness)
        ws_clipped = np.clip(grid_5ch[4], 0, 19.5).ravel()
        pts_ws = np.stack([ws_clipped, d_flat], axis=-1)
        reduction_ws = self._friction_interp(pts_ws).reshape(grid_size, grid_size)
        adjusted[4] = np.maximum(0.0, adjusted[4] - reduction_ws)
        
        return adjusted


if __name__ == "__main__":
    print("Initializing Multi-Variable Urban Physics Engine...")
    uhi = UHIFuzzyAdjuster()
    
    # Sanity test
    test_frame = np.stack([
        np.full((17, 17), 40.0),
        np.full((17, 17), 0.5),
        np.full((17, 17), 1013.0),
        np.full((17, 17), 60.0),
        np.full((17, 17), 7.0),
    ], axis=0)

    adjusted = uhi.adjust_full_grid(test_frame)

    print("\nSanity Check Results (Dubai Core pixel [12,10]):")
    print(f"  T_avg: {test_frame[0,12,10]:.1f} --> {adjusted[0,12,10]:.1f} C   (delta: +{adjusted[0,12,10]-test_frame[0,12,10]:.2f})")
    print(f"  PCP:   {test_frame[1,12,10]:.1f} --> {adjusted[1,12,10]:.1f} mm  (delta: {adjusted[1,12,10]-test_frame[1,12,10]:.2f}) [UNTOUCHED]")
    print(f"  AP:    {test_frame[2,12,10]:.1f} --> {adjusted[2,12,10]:.1f} hPa (delta: {adjusted[2,12,10]-test_frame[2,12,10]:.2f}) [UNTOUCHED]")
    print(f"  RH:    {test_frame[3,12,10]:.1f} --> {adjusted[3,12,10]:.1f} %   (delta: {adjusted[3,12,10]-test_frame[3,12,10]:.2f})")
    print(f"  WS:    {test_frame[4,12,10]:.1f} --> {adjusted[4,12,10]:.1f} m/s (delta: {adjusted[4,12,10]-test_frame[4,12,10]:.2f})")

    print("\nDesert pixel [2,2]:")
    print(f"  T_avg: {test_frame[0,2,2]:.1f} --> {adjusted[0,2,2]:.1f} C   (delta: +{adjusted[0,2,2]-test_frame[0,2,2]:.2f})")
    print(f"  RH:    {test_frame[3,2,2]:.1f} --> {adjusted[3,2,2]:.1f} %   (delta: {adjusted[3,2,2]-test_frame[3,2,2]:.2f})")
    print(f"  WS:    {test_frame[4,2,2]:.1f} --> {adjusted[4,2,2]:.1f} m/s (delta: {adjusted[4,2,2]-test_frame[4,2,2]:.2f})")

    # Speed benchmark
    import time
    n_iter = 1000
    t0 = time.time()
    for _ in range(n_iter):
        uhi.adjust_full_grid(test_frame)
    elapsed = time.time() - t0
    print(f"\nBenchmark: {n_iter} grids in {elapsed:.2f}s = {n_iter/elapsed:.0f} grids/sec")
    print(f"  Estimated time for 31,400 grids: {31400 / (n_iter/elapsed) / 60:.1f} minutes")
