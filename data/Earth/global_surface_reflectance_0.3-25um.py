"""
Global-Average Spectral Surface Albedo/Reflectance of Earth
Full range: 0.3 – 25 µm

Short-wave (0.3–4 µm): True surface albedo (reflected solar radiation)
Mid-wave (4–5 µm): Transition zone (mix of reflected solar + thermal emission)
Thermal IR (5–25 µm): Spectral reflectance = 1 - emissivity

Sources:
  - Bowker et al. (1985), "Spectral Reflectances of Natural Targets"
  - Baldridge et al. (2009), ASTER Spectral Library v2.0
  - USGS Spectral Library (Kokaly et al., 2017)
  - Morel (1988), ocean reflectance
  - Warren (1982), snow/ice optical properties
  - Warren & Brandt (2008), updated ice optical constants
  - Salisbury & D'Aria (1992), emissivity of terrestrial materials
  - Snyder et al. (1998), MODIS land surface emissivity
  - Hulley et al. (2015), ECOSTRESS spectral library
  - Masuda et al. (1988), ocean emissivity
  - Downing & Williams (1975), optical constants of water
  - Christensen et al. (2000), far-IR mineral emissivity (TES/Mars context,
    but applicable to terrestrial silicates)

Note: In the thermal IR, "reflectance" = 1 - emissivity. Beyond ~15 µm,
data becomes sparse and values are more approximate.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================
# Wavelength grid (nm) — 300 nm to 25000 nm (0.3–25 µm)
# ============================================================
wavelengths = np.array([
    # UV-Vis-NIR (300–1000 nm)
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
    800, 850, 900, 950, 1000,
    # SWIR (1000–2500 nm)
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2100, 2200, 2300, 2400, 2500,
    # Mid-wave IR transition (2500–5000 nm)
    2600, 2800, 3000, 3500, 4000, 4500, 5000,
    # Thermal IR (5000–15000 nm)
    5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000,
    10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000,
    # Far-IR (15000–25000 nm)
    15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000,
    21000, 22000, 23000, 24000, 25000
])

n = len(wavelengths)

# ============================================================
# OCEAN
# ============================================================
# SW: Morel (1988)
# TIR 5–15 µm: Masuda et al. (1988), ε ≈ 0.98–0.99
# Far-IR 15–25 µm: Downing & Williams (1975) optical constants of water
#   Water emissivity remains very high (~0.98–0.99) throughout far-IR,
#   with minor features. Slight decrease in ε near 17–22 µm due to
#   librational band shoulder, but still >0.97.
ocean = np.array([
    # UV-Vis-NIR
    0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02,
    0.02, 0.02, 0.02, 0.02, 0.02,
    # SWIR
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01,
    # Mid-wave transition
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    # Thermal IR 5–15 µm
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02,
    # Far-IR 15–25 µm
    0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.01, 0.01
])

# ============================================================
# FOREST (deciduous/mixed)
# ============================================================
# TIR: ε ≈ 0.97–0.99, cellulose feature near 9–10 µm
# Far-IR: vegetation emissivity stays high (>0.97), dominated by
#   water content in leaves. Dry wood has some minor features
#   near 16–18 µm but living vegetation is close to blackbody.
vegetation_forest = np.array([
    # UV-Vis-NIR
    0.02, 0.02, 0.03, 0.04, 0.06, 0.08, 0.05, 0.04, 0.05, 0.30,
    0.40, 0.42, 0.40, 0.38, 0.35,
    # SWIR
    0.30, 0.25, 0.22, 0.10, 0.18, 0.22, 0.20, 0.12, 0.08, 0.15,
    0.14, 0.10, 0.08, 0.06, 0.05,
    # Mid-wave transition
    0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02,
    # Thermal IR 5–15 µm
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.04, 0.03,
    0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02,
    # Far-IR 15–25 µm
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01
])

# ============================================================
# GRASSLAND / CROPS
# ============================================================
grassland = np.array([
    # UV-Vis-NIR
    0.03, 0.03, 0.04, 0.05, 0.08, 0.12, 0.08, 0.06, 0.06, 0.35,
    0.45, 0.47, 0.45, 0.42, 0.40,
    # SWIR
    0.35, 0.30, 0.26, 0.12, 0.22, 0.26, 0.24, 0.14, 0.10, 0.18,
    0.16, 0.12, 0.10, 0.08, 0.06,
    # Mid-wave transition
    0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02,
    # Thermal IR 5–15 µm
    0.02, 0.02, 0.03, 0.03, 0.02, 0.02, 0.04, 0.05, 0.05, 0.03,
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03,
    # Far-IR 15–25 µm
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    0.01, 0.01, 0.01, 0.01, 0.01
])

# ============================================================
# DESERT / BARE ARID SOIL
# ============================================================
# This is the most spectrally interesting surface in the far-IR.
# Quartz restrahlen bands: ~8.2 µm (ν3 asymm. stretch), ~9.0 µm
# Additional silicate features:
#   - ~12.5 µm: Si-O bending mode
#   - ~16–18 µm: secondary restrahlen / lattice modes in feldspars
#     and clays; ε can dip to ~0.85–0.90
#   - ~20–22 µm: further lattice vibrations, ε recovers toward 0.92–0.95
#   - Beyond 23 µm: ε → 0.95+ for most silicates
# Based on Salisbury & D'Aria (1992), Christensen et al. (2000),
# and ECOSTRESS spectral library data.
desert = np.array([
    # UV-Vis-NIR
    0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.33,
    0.35, 0.36, 0.37, 0.38, 0.39,
    # SWIR
    0.40, 0.42, 0.43, 0.30, 0.42, 0.44, 0.44, 0.42, 0.38, 0.40,
    0.39, 0.37, 0.35, 0.33, 0.30,
    # Mid-wave transition
    0.28, 0.25, 0.22, 0.18, 0.15, 0.12, 0.08,
    # Thermal IR 5–15 µm (quartz restrahlen)
    0.06, 0.05, 0.05, 0.06, 0.06, 0.20, 0.28, 0.30, 0.22, 0.08,
    0.06, 0.05, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07,
    # Far-IR 15–25 µm (secondary silicate features)
    0.08, 0.09, 0.10, 0.12, 0.13, 0.14, 0.13, 0.12, 0.10, 0.09,
    0.08, 0.07, 0.06, 0.05, 0.05
])

# ============================================================
# SNOW / ICE
# ============================================================
# TIR: ε ≈ 0.98–0.99 in 8–14 µm window
# Far-IR: Warren & Brandt (2008) — ice absorption remains strong.
#   ε stays ≈ 0.98–0.99 throughout 15–25 µm. There is a broad
#   lattice vibration absorption near 44 µm (well beyond our range),
#   but in 15–25 µm ice remains an excellent absorber/emitter.
snow_ice = np.array([
    # UV-Vis-NIR
    0.95, 0.95, 0.95, 0.97, 0.97, 0.96, 0.95, 0.94, 0.92, 0.88,
    0.82, 0.75, 0.60, 0.55, 0.50,
    # SWIR
    0.40, 0.25, 0.20, 0.05, 0.12, 0.10, 0.08, 0.04, 0.02, 0.05,
    0.04, 0.03, 0.02, 0.02, 0.01,
    # Mid-wave transition
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    # Thermal IR 5–15 µm
    0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02,
    # Far-IR 15–25 µm
    0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01
])

# ============================================================
# TUNDRA
# ============================================================
tundra = np.array([
    # UV-Vis-NIR
    0.04, 0.04, 0.05, 0.06, 0.08, 0.10, 0.08, 0.07, 0.08, 0.25,
    0.30, 0.32, 0.30, 0.28, 0.26,
    # SWIR
    0.23, 0.20, 0.18, 0.08, 0.15, 0.18, 0.16, 0.10, 0.07, 0.12,
    0.11, 0.09, 0.07, 0.05, 0.04,
    # Mid-wave transition
    0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02,
    # Thermal IR 5–15 µm
    0.03, 0.03, 0.03, 0.03, 0.02, 0.03, 0.04, 0.04, 0.04, 0.03,
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03,
    # Far-IR 15–25 µm (mix of soil mineral features + vegetation)
    0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03,
    0.02, 0.02, 0.02, 0.02, 0.02
])

# ============================================================
# URBAN
# ============================================================
# Concrete and asphalt have some silicate-related features in far-IR
# but generally ε ≈ 0.92–0.96
urban = np.array([
    # UV-Vis-NIR
    0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.17, 0.18, 0.19, 0.20,
    0.21, 0.22, 0.22, 0.22, 0.22,
    # SWIR
    0.22, 0.22, 0.22, 0.15, 0.20, 0.21, 0.20, 0.18, 0.16, 0.18,
    0.17, 0.16, 0.15, 0.14, 0.13,
    # Mid-wave transition
    0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,
    # Thermal IR 5–15 µm
    0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.07, 0.05,
    0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06,
    # Far-IR 15–25 µm
    0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05,
    0.04, 0.04, 0.04, 0.04, 0.04
])

# ============================================================
# WETLAND / INLAND WATER
# ============================================================
wetland = np.array([
    # UV-Vis-NIR
    0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04,
    0.03, 0.03, 0.03, 0.03, 0.02,
    # SWIR
    0.02, 0.02, 0.02, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01,
    # Mid-wave transition
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    # Thermal IR 5–15 µm
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02,
    # Far-IR 15–25 µm (similar to ocean)
    0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.01, 0.01
])

# ============================================================
# Area fractions
# ============================================================
area_fractions = {
    'Ocean':            0.708,
    'Forest':           0.100,
    'Grassland/Crops':  0.070,
    'Desert/Bare Soil': 0.040,
    'Snow/Ice':         0.035,
    'Tundra':           0.020,
    'Urban':            0.012,
    'Wetland/Water':    0.015,
}

spectra = {
    'Ocean':            ocean,
    'Forest':           vegetation_forest,
    'Grassland/Crops':  grassland,
    'Desert/Bare Soil': desert,
    'Snow/Ice':         snow_ice,
    'Tundra':           tundra,
    'Urban':            urban,
    'Wetland/Water':    wetland,
}

# Verify lengths
for name, spec in spectra.items():
    assert len(spec) == n, f"{name} has {len(spec)} values, expected {n}"

total_frac = sum(area_fractions.values())
print(f"Total area fraction: {total_frac:.3f}")

# ============================================================
# Compute global-average spectral surface reflectance
# ============================================================
global_refl = np.zeros(n, dtype=float)
for stype, spectrum in spectra.items():
    global_refl += area_fractions[stype] * spectrum

# Also compute global emissivity in thermal range
global_emissivity = 1.0 - global_refl

# ============================================================
# Print table
# ============================================================
print(f"\n{'λ (nm)':>10} | {'λ (µm)':>8} | {'Regime':>14} | {'Reflectance':>12} | {'Emissivity':>11}")
print("-" * 72)
for i, wl in enumerate(wavelengths):
    wl_um = wl / 1000.0
    if wl <= 4000:
        regime = "Solar"
    elif wl <= 5000:
        regime = "Transition"
    else:
        regime = "Thermal IR"
    print(f"  {wl:>8.0f} | {wl_um:>7.2f}  | {regime:>14} |   {global_refl[i]:.4f}     | {global_emissivity[i]:.4f}")

# Solar-weighted broadband
def solar_weight(wl_nm):
    h, c, k, T = 6.626e-34, 3e8, 1.381e-23, 5778.0
    wl_m = wl_nm * 1e-9
    intensity = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * T)) - 1)
    return intensity

sw_idx = np.where(wavelengths <= 4000)[0]
solar_w = np.array([solar_weight(wavelengths[i]) for i in sw_idx])
solar_w /= solar_w.sum()
bb_albedo_solar = np.sum(global_refl[sw_idx] * solar_w)
print(f"\nSolar-weighted SW mean albedo (0.3–4 µm): {bb_albedo_solar:.4f}")

tir_idx = np.where(wavelengths > 5000)[0]
mean_tir_refl = np.mean(global_refl[tir_idx])
mean_tir_emiss = np.mean(global_emissivity[tir_idx])
print(f"Mean thermal IR reflectance (5–25 µm): {mean_tir_refl:.4f}")
print(f"Mean thermal IR emissivity  (5–25 µm): {mean_tir_emiss:.4f}")

fir_idx = np.where(wavelengths > 15000)[0]
mean_fir_refl = np.mean(global_refl[fir_idx])
mean_fir_emiss = np.mean(global_emissivity[fir_idx])
print(f"Mean far-IR reflectance (15–25 µm): {mean_fir_refl:.4f}")
print(f"Mean far-IR emissivity  (15–25 µm): {mean_fir_emiss:.4f}")

# ============================================================
# PLOT — 4-panel figure
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 18))

wl_um = wavelengths / 1000.0

colors = {
    'Ocean': '#1f77b4',
    'Forest': '#2ca02c',
    'Grassland/Crops': '#8bc34a',
    'Desert/Bare Soil': '#d2691e',
    'Snow/Ice': '#87ceeb',
    'Tundra': '#9e9e9e',
    'Urban': '#555555',
    'Wetland/Water': '#00bcd4',
}

# --- Panel 1: All surface types, full range ---
ax1 = axes[0]
for stype, spectrum in spectra.items():
    ax1.plot(wl_um, spectrum, '-o', markersize=2, color=colors[stype],
             label=f"{stype} ({area_fractions[stype]*100:.1f}%)", linewidth=1.1)

ax1.axvline(4.0, color='red', linestyle=':', alpha=0.5)
ax1.axvline(5.0, color='red', linestyle=':', alpha=0.5)
ax1.axvline(15.0, color='purple', linestyle=':', alpha=0.4)
ax1.text(2.0, 0.92, 'Solar reflectance', fontsize=9, ha='center', color='#555')
ax1.text(4.5, 0.92, 'T', fontsize=8, ha='center', color='red', alpha=0.7)
ax1.text(10.0, 0.92, '1 − emissivity', fontsize=9, ha='center', color='#555')
ax1.text(20.0, 0.92, 'Far-IR', fontsize=9, ha='center', color='purple', alpha=0.7)

ax1.set_xlabel('Wavelength (µm)', fontsize=12)
ax1.set_ylabel('Surface Reflectance', fontsize=12)
ax1.set_title('Spectral Surface Reflectance by Surface Type (0.3–25 µm)', fontsize=14)
ax1.legend(fontsize=7.5, loc='upper right', ncol=2)
ax1.set_xlim(0.3, 25)
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Global average, full range ---
ax2 = axes[1]
ax2.plot(wl_um, global_refl, '-o', markersize=2.5, color='#333333',
         linewidth=2, label='Global-average surface reflectance')
ax2.fill_between(wl_um, 0, global_refl, alpha=0.12, color='#333333')

ax2.axvline(4.0, color='red', linestyle=':', alpha=0.5)
ax2.axvline(5.0, color='red', linestyle=':', alpha=0.5)
ax2.axvline(15.0, color='purple', linestyle=':', alpha=0.4)

ax2.set_xlabel('Wavelength (µm)', fontsize=12)
ax2.set_ylabel('Surface Reflectance', fontsize=12)
ax2.set_title('Global-Average Spectral Surface Reflectance (0.3–25 µm)', fontsize=14)
ax2.legend(fontsize=10)
ax2.set_xlim(0.3, 25)
ax2.set_ylim(0, 0.16)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Thermal + Far-IR detail (5–25 µm), reflectance ---
ax3 = axes[2]
tir_full_mask = wavelengths > 5000
tir_wl = wl_um[tir_full_mask]
tir_refl = global_refl[tir_full_mask]

ax3.plot(tir_wl, tir_refl, '-o', markersize=4, color='#c62828',
         linewidth=2, label='Global avg reflectance (1 − ε)')
ax3.fill_between(tir_wl, 0, tir_refl, alpha=0.12, color='#c62828')

# Desert contribution
desert_contrib = area_fractions['Desert/Bare Soil'] * desert[tir_full_mask]
ax3.plot(tir_wl, desert_contrib, '--', color='#d2691e', linewidth=1.2,
         label='Desert contribution (4%)', alpha=0.8)

ax3.axvline(15.0, color='purple', linestyle=':', alpha=0.4)
ax3.text(10, 0.033, 'Standard TIR window', fontsize=8, ha='center', color='#555')
ax3.text(20, 0.033, 'Far-IR', fontsize=8, ha='center', color='purple', alpha=0.7)

ax3.annotate('Quartz\nrestrahlen',
             xy=(8.75, global_refl[wavelengths == 8500][0]),
             xytext=(7, 0.028),
             fontsize=9, color='#d2691e',
             arrowprops=dict(arrowstyle='->', color='#d2691e', lw=1.2))

ax3.annotate('Secondary\nsilicate\nfeatures',
             xy=(17.5, global_refl[wavelengths == 17500][0]),
             xytext=(22, 0.028),
             fontsize=9, color='#d2691e',
             arrowprops=dict(arrowstyle='->', color='#d2691e', lw=1.2))

ax3.set_xlabel('Wavelength (µm)', fontsize=12)
ax3.set_ylabel('Reflectance (1 − ε)', fontsize=12)
ax3.set_title('Thermal + Far-IR: Global-Average Surface Reflectance (5–25 µm)', fontsize=14)
ax3.legend(fontsize=9)
ax3.set_xlim(5, 25)
ax3.set_ylim(0, 0.036)
ax3.grid(True, alpha=0.3)

# --- Panel 4: Global emissivity (5–25 µm) ---
ax4 = axes[3]
tir_emiss = global_emissivity[tir_full_mask]
ax4.plot(tir_wl, tir_emiss, '-o', markersize=4, color='#1565c0',
         linewidth=2, label='Global avg emissivity')
ax4.fill_between(tir_wl, tir_emiss, 1.0, alpha=0.08, color='#1565c0')

ax4.axvline(15.0, color='purple', linestyle=':', alpha=0.4)
ax4.axhline(1.0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

ax4.set_xlabel('Wavelength (µm)', fontsize=12)
ax4.set_ylabel('Emissivity', fontsize=12)
ax4.set_title('Global-Average Surface Emissivity (5–25 µm)', fontsize=14)
ax4.legend(fontsize=10)
ax4.set_xlim(5, 25)
ax4.set_ylim(0.960, 1.001)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# plt.savefig('/home/claude/global_surface_reflectance_0.3-25um.png', dpi=150, bbox_inches='tight')
# print("\nPlot saved.")
#
# # ============================================================
# # Save CSV
# # ============================================================
# csv_path = '/home/claude/global_surface_reflectance_0.3-25um.csv'
# with open(csv_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     header = ['wavelength_nm', 'wavelength_um', 'regime',
#               'global_avg_reflectance', 'global_avg_emissivity'] + list(spectra.keys())
#     writer.writerow(header)
#     for i, wl in enumerate(wavelengths):
#         wl_um_val = wl / 1000.0
#         if wl <= 4000:
#             regime = 'solar_reflectance'
#         elif wl <= 5000:
#             regime = 'transition'
#         else:
#             regime = 'thermal_IR_1-emissivity'
#         row = [wl, f"{wl_um_val:.3f}", regime,
#                f"{global_refl[i]:.5f}", f"{global_emissivity[i]:.5f}"]
#         for stype in spectra:
#             row.append(f"{spectra[stype][i]:.4f}")
#         writer.writerow(row)
#
# print(f"CSV saved to {csv_path}")
# print("\nDone!")
