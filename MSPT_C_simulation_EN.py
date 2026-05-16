"""
Mechanical Spiral-Phased Transducer (MSPT):
A Geometric Principle for Deterministic Control of Travelling Boundary Conditions, Wave Forcing and Vortical Flows
Preprint for Zenodo
Author: Vladimir I. Khaustov
Independent Researcher
ORCID: 0009-0007-3657-2309
DOI: 10.5281/zenodo.20237780
Licence: CC-BY 4.0
Project website: https://vihrihaosa.ru/
Publication date: 16 May 2026
"""


"""
MSPT-C simulation — cylindrical configuration.

This script visualises two coaxial cylinders:
  - inner cylinder A carries a helical aperture array;
  - outer cylinder B carries axially inclined shutters;
  - the travelling point of aperture-shutter coincidence propagates
    along the z-axis.

The script produces:
  1. a 3D visualisation of the cylindrical MSPT geometry;
  2. an unrolled developed-surface representation;
  3. a set of travelling-wave frames during one angular increment.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; required by some Matplotlib backends
from matplotlib.patches import Circle

# ============================================================
# DESIGN PARAMETERS
# ============================================================

# --- Cylinder geometry ---
R_A      = 0.050        # inner cylinder radius, m  (50 mm)
delta    = 0.002        # working gap, m            (2 mm)
R_B      = R_A + delta  # outer cylinder radius
L_cyl    = 0.200        # cylinder length, m        (200 mm)

# --- Helical aperture array on cylinder A ---
N_turns  = 8            # number of helical turns
N_pt     = 30           # apertures per turn
N_total  = N_turns * N_pt   # 240 apertures total
d_ap     = 0.004        # aperture diameter, m

# --- Shutter family on cylinder B ---
N_shut   = 30           # number of shutters; matches N_pt

# --- Derived quantities ---
dphi_ap  = 2*np.pi / N_pt           # angular pitch = 12 degrees
dz_turn  = L_cyl / N_turns          # axial pitch per turn = 25 mm
R_mean   = 0.5*(R_A + R_B)

# --- Operating point ---
f_rot    = 50.0                     # rotation frequency, rev/s

# ============================================================
# KINEMATIC PREDICTIONS
# ============================================================

# Time for one angular increment of dphi_ap.
dt_step = dphi_ap / (2*np.pi * f_rot)

# Axial sweep velocity: the open coincidence point moves along z.
V_axial = L_cyl / dt_step

# Helical arc length per sweep: length of the full helix on the developed surface.
L_helix = N_turns * np.sqrt((2*np.pi*R_mean)**2 + dz_turn**2)

# Velocity along the helix.
V_helix = L_helix / dt_step

# Surface velocity of cylinder A at radius R_A.
V_surf  = 2*np.pi * R_A * f_rot

print("="*60)
print("CYLINDRICAL MSPT — KINEMATIC PREDICTIONS")
print("="*60)
print(f"  R_A         = {R_A*1000:.1f} mm")
print(f"  R_B         = {R_B*1000:.1f} mm")
print(f"  L_cyl       = {L_cyl*1000:.1f} mm")
print(f"  N_turns     = {N_turns}")
print(f"  N_pt        = {N_pt}")
print(f"  N_total     = {N_total} apertures")
print(f"  dz_turn     = {dz_turn*1000:.2f} mm")
print(f"  dphi_ap     = {np.degrees(dphi_ap):.1f} deg")
print("-"*60)
print(f"  f_rot       = {f_rot:.1f} rev/s")
print(f"  dt_step     = {dt_step*1e6:.1f} us")
print(f"  V_surf      = {V_surf:.2f} m/s   (cylinder surface)")
print(f"  V_axial     = {V_axial:.2f} m/s   (z-sweep of open point)")
print(f"  V_helix     = {V_helix:.2f} m/s   (along developed helix)")
print(f"  K_axial     = {V_axial/V_surf:.2f}")
print(f"  K_helix     = {V_helix/V_surf:.2f}")
print("="*60)

# ============================================================
# APERTURE POSITIONS
# ============================================================

# Parametrise the helix by angular variable phi from 0 to 2*pi*N_turns.
phi_ap = np.linspace(0, 2*np.pi*N_turns, N_total, endpoint=False)
z_ap   = L_cyl * phi_ap / (2*np.pi*N_turns)
x_ap   = R_A * np.cos(phi_ap)
y_ap   = R_A * np.sin(phi_ap)

# ============================================================
# SHUTTER CURVES
# ============================================================
# Each shutter extends from z=0 to z=L_cyl and has an angular offset
# of dphi_ap between the lower and upper endpoints:
#
#     phi_sh(z) = alpha_0 - dphi_ap * (z / L_cyl)

z_grid = np.linspace(0, L_cyl, 60)
shutter_alphas = np.linspace(0, 2*np.pi, N_shut, endpoint=False)

# ============================================================
# 3D VISUALISATION
# ============================================================

fig = plt.figure(figsize=(14, 9))

# ----- Left subplot: full 3D view -----
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# Draw cylinder A surface as a semi-transparent surface.
theta = np.linspace(0, 2*np.pi, 80)
Z, T  = np.meshgrid(np.linspace(0, L_cyl, 30), theta)
X_A   = R_A * np.cos(T)
Y_A   = R_A * np.sin(T)
ax1.plot_surface(X_A, Y_A, Z, color='steelblue',
                 alpha=0.12, linewidth=0, antialiased=True)

# Draw cylinder B surface as a very faint outer surface.
X_B = R_B * np.cos(T)
Y_B = R_B * np.sin(T)
ax1.plot_surface(X_B, Y_B, Z, color='firebrick',
                 alpha=0.05, linewidth=0, antialiased=True)

# Apertures on cylinder A.
ax1.scatter(x_ap, y_ap, z_ap, s=22,
            c='darkblue', depthshade=True,
            label=f'{N_total} apertures (helix)')

# Highlight the helix curve for clarity.
phi_h = np.linspace(0, 2*np.pi*N_turns, 1000)
z_h   = L_cyl * phi_h / (2*np.pi*N_turns)
ax1.plot(R_A*np.cos(phi_h), R_A*np.sin(phi_h), z_h,
         color='lightsteelblue', linewidth=0.8,
         linestyle='--', alpha=0.7)

# Shutters on cylinder B.
for alpha_0 in shutter_alphas:
    phi_sh = alpha_0 - dphi_ap * (z_grid / L_cyl)
    xs = R_B * np.cos(phi_sh)
    ys = R_B * np.sin(phi_sh)
    ax1.plot(xs, ys, z_grid, color='darkred',
             linewidth=1.0, alpha=0.8)

# Axis of rotation.
ax1.plot([0, 0], [0, 0], [-0.02, L_cyl + 0.02],
         color='black', linewidth=1.5)

ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Cylindrical MSPT — 3D view\n'
              f'(Inner R_A = {R_A*1000:.0f} mm, '
              f'L = {L_cyl*1000:.0f} mm, '
              f'{N_turns} turns × {N_pt} apertures)',
              fontsize=10)
ax1.view_init(elev=15, azim=-55)
ax1.set_box_aspect([1, 1, 2.2])

# ----- Right subplot: unrolled developed surface -----
ax2 = fig.add_subplot(1, 2, 2)

# Show apertures as circles on the (phi, z) plane.
for i in range(N_total):
    circ = Circle((np.degrees(phi_ap[i] % (2*np.pi)), z_ap[i]*1000),
                  radius=2.0, facecolor='darkblue',
                  edgecolor='black', linewidth=0.3)
    ax2.add_patch(circ)

# Shutters as inclined lines on the (phi, z) plane.
for alpha_0 in shutter_alphas:
    phi_sh = alpha_0 - dphi_ap * (z_grid / L_cyl)
    phi_sh_deg = np.degrees(phi_sh % (2*np.pi))

    # Break lines where they wrap around the 0/360-degree boundary.
    breaks = np.where(np.abs(np.diff(phi_sh_deg)) > 180)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends   = np.concatenate((breaks + 1, [len(phi_sh_deg)]))
    for s, e in zip(starts, ends):
        ax2.plot(phi_sh_deg[s:e], z_grid[s:e]*1000,
                 color='darkred', linewidth=1.2, alpha=0.7)

ax2.set_xlim(0, 360)
ax2.set_ylim(0, L_cyl*1000)
ax2.set_xlabel('Angular position phi (degrees)', fontsize=11)
ax2.set_ylabel('Axial position z (mm)', fontsize=11)
ax2.set_title('Unrolled cylindrical surface\n'
              '(apertures: blue dots, shutters: red lines)',
              fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('auto')

# Annotation on the developed view.
ax2.annotate('dphi_ap = 12°\nper shutter',
             xy=(180, 100), xytext=(220, 50),
             fontsize=9, color='darkred',
             arrowprops=dict(arrowstyle='->', color='darkred'))

plt.suptitle('Fig. 6. Cylindrical Spiral-Phased Transducer (MSPT-C)',
             fontsize=13, y=1.00)

plt.tight_layout()
plt.savefig('MSPT_C_cylindrical.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# ADDITIONAL FIGURE: TRAVELLING WAVE FRAMES
# ============================================================

fig2, axes = plt.subplots(1, 4, figsize=(20, 5.5))

# Determine which axial layer each aperture belongs to.
# Apertures are grouped into N_turns layers by angular bin.
layer_index = (phi_ap // (2*np.pi)).astype(int)

# Show four frames during one angular increment dphi_ap.
n_frames = 4

for k, ax in enumerate(axes):
    # All apertures projected to the (phi, z) plane.
    phi_deg = np.degrees(phi_ap % (2*np.pi))
    ax.scatter(phi_deg, z_ap*1000, s=15,
               c='lightgray', edgecolor='gray', linewidth=0.3)

    # Highlight one layer of apertures as open.
    open_idx = np.where(layer_index == k * (N_turns // n_frames))[0]
    ax.scatter(phi_deg[open_idx], z_ap[open_idx]*1000, s=80,
               c='red', edgecolor='darkred', linewidth=0.5,
               zorder=5)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, L_cyl*1000)
    ax.set_xlabel('phi (deg)')
    if k == 0:
        ax.set_ylabel('z (mm)')
    ax.set_title(f't = {k} * dt / 4', fontsize=11)
    ax.grid(True, alpha=0.3)

fig2.suptitle('Fig. 7. Cylindrical MSPT — axial sweep of the '
              'open-aperture pattern\n'
              'during one angular increment dphi_ap '
              '(developed surface representation)',
              fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig('MSPT_C_axial_sweep.png', dpi=300, bbox_inches='tight')
plt.show()
