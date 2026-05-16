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
MSPT-D simulation — disk configuration.

This script visualises the operation of a two-disk valve mechanism:
  - disk A contains a four-turn Archimedean spiral aperture array;
  - disk B contains 30 inclined shutters;
  - the coincidence between apertures and shutters forms a travelling
    pattern of open apertures.

The script produces:
  1. a static three-panel geometry figure;
  2. a radius-versus-time plot of the travelling coincidence point;
  3. an animation of the rotating shutter disk relative to the aperture disk.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# MSPT-D PARAMETERS
# ============================================================
D_DISK = 150.0           # disk diameter, mm
R_INNER = 55.0           # inner spiral radius, mm (inner diameter 110 mm)
R_OUTER = 70.0           # outer spiral radius, mm (outer diameter 140 mm)
N_TURNS = 4              # number of spiral turns
N_HOLES_PER_TURN = 30    # apertures per turn
HOLE_DIAM = 4.0          # aperture diameter, mm
N_HOLES_TOTAL = N_TURNS * N_HOLES_PER_TURN  # 120 apertures
N_SHUTTERS = N_HOLES_PER_TURN               # 30 shutters
SHUTTER_WIDTH = 4.0      # shutter width, mm

# Angular pitch between apertures on one turn.
DPHI_HOLE = 2 * np.pi / N_HOLES_PER_TURN  # 12 degrees

# Radial pitch of the spiral per turn.
DR_TURN = (R_OUTER - R_INNER) / N_TURNS  # 3.75 mm per turn

# Rotation frequency used for the kinematic prediction.
F_ROTATION = 50.0  # rev/s (3000 rpm)

# ============================================================
# DISK A GEOMETRY — SPIRAL APERTURE ARRAY
# ============================================================
def make_spiral_holes():
    """
    Generate the aperture centres on an Archimedean spiral.

    Spiral equation:
        r(phi) = R_INNER + (DR_TURN / (2*pi)) * phi

    The angular variable phi runs from 0 to N_TURNS * 2*pi.
    Apertures are uniformly distributed by angular pitch DPHI_HOLE.
    """
    holes = []
    for i in range(N_HOLES_TOTAL):
        phi = i * DPHI_HOLE
        r = R_INNER + (DR_TURN / (2 * np.pi)) * phi
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        holes.append((x, y, r, phi, i))
    return holes

# ============================================================
# DISK B GEOMETRY — INCLINED SHUTTERS
# ============================================================
def make_shutters(rotation_angle=0.0):
    """
    Generate 30 inclined shutter curves.

    Each shutter is a curve extending from R_INNER to R_OUTER.
    Its inner endpoint is located at angular coordinate alpha_0.
    Its outer endpoint is shifted by one aperture pitch, alpha_0 - DPHI_HOLE.

    This angular inclination ensures that, when disk B is rotated by one
    angular pitch DPHI_HOLE, the coincidence point sweeps from R_INNER
    to R_OUTER and sequentially activates the four spiral turns.

    Radial parametrisation:
        phi_shutter(r) = alpha_0 - DPHI_HOLE * (r - R_INNER) / (R_OUTER - R_INNER)
    """
    shutters = []
    for k in range(N_SHUTTERS):
        alpha_0 = k * (2 * np.pi / N_SHUTTERS) + rotation_angle
        r_arr = np.linspace(R_INNER, R_OUTER, 50)
        phi_arr = alpha_0 - DPHI_HOLE * (r_arr - R_INNER) / (R_OUTER - R_INNER)
        x_arr = r_arr * np.cos(phi_arr)
        y_arr = r_arr * np.sin(phi_arr)
        shutters.append((x_arr, y_arr, alpha_0))
    return shutters

# ============================================================
# COINCIDENCE DETECTION
# ============================================================
def find_open_holes(holes, rotation_angle, tolerance_rad=None):
    """
    Determine which apertures are open.

    An aperture is treated as open when a shutter passes over it.
    For a shutter k at radius r:

        phi_k(r) = k*(2*pi/N_SHUTTERS) + rotation
                   - DPHI_HOLE*(r - R_INNER)/(R_OUTER - R_INNER)

    An aperture at (r_h, phi_h) is open if, for at least one shutter k,
    the angular mismatch is smaller than the specified tolerance.
    """
    if tolerance_rad is None:
        # Conservative angular half-width of an aperture at the inner radius.
        tolerance_rad = (HOLE_DIAM / 2) / R_INNER

    open_list = []
    for (x, y, r, phi, idx) in holes:
        for k in range(N_SHUTTERS):
            phi_k = (
                k * (2 * np.pi / N_SHUTTERS)
                + rotation_angle
                - DPHI_HOLE * (r - R_INNER) / (R_OUTER - R_INNER)
            )
            # Angular difference wrapped to [-pi, pi].
            d = (phi - phi_k + np.pi) % (2 * np.pi) - np.pi
            if abs(d) < tolerance_rad:
                open_list.append((x, y, r, phi, idx))
                break
    return open_list

# ============================================================
# KINEMATIC PARAMETERS
# ============================================================
def print_kinematics():
    """Print the main kinematic quantities for the disk MSPT."""
    omega = 2 * np.pi * F_ROTATION  # rad/s
    v_edge = omega * (R_OUTER / 1000.0)  # m/s, linear speed at the outer radius

    # Travelling point: during one DPHI_HOLE rotation, the pattern sweeps the full spiral.
    # Approximate spiral length: mean circumference multiplied by the number of turns.
    R_mean = (R_INNER + R_OUTER) / 2.0 / 1000.0  # m
    L_spiral = 2 * np.pi * R_mean * N_TURNS  # m

    # Time for one angular pitch.
    t_step = DPHI_HOLE / omega  # s

    # Effective travelling-pattern velocity along the spiral.
    v_wave = L_spiral / t_step  # m/s

    # Number of full spiral sweeps per second.
    f_runs_per_sec = N_SHUTTERS * F_ROTATION

    print("=" * 60)
    print("MSPT-D KINEMATICS")
    print("=" * 60)
    print(f"Disk diameter:                   {D_DISK:.1f} mm")
    print(f"Spiral radius range:             R = {R_INNER}..{R_OUTER} mm")
    print(f"Number of turns:                 {N_TURNS}")
    print(f"Apertures per turn:              {N_HOLES_PER_TURN}")
    print(f"Total apertures:                 {N_HOLES_TOTAL}")
    print(f"Number of shutters:              {N_SHUTTERS}")
    print(f"Angular aperture pitch:          {np.degrees(DPHI_HOLE):.2f} deg")
    print(f"Radial pitch per turn:           {DR_TURN:.3f} mm")
    print("-" * 60)
    print(f"Rotation frequency:              {F_ROTATION} rev/s ({F_ROTATION*60:.0f} rpm)")
    print(f"Linear edge speed:               {v_edge:.2f} m/s")
    print("-" * 60)
    print(f"Mean-radius spiral length:       {L_spiral*1000:.1f} mm")
    print(f"Time for one sweep:              {t_step*1e6:.2f} microseconds")
    print(f"Travelling-pattern velocity:     {v_wave:.1f} m/s")
    print(f"Mach number in air, a=340 m/s:   {v_wave/340:.2f}")
    print(f"Mach number in water, a=1500 m/s:{v_wave/1500:.2f}")
    print(f"Spiral sweeps per second:        {f_runs_per_sec:.0f} Hz")
    print("=" * 60)
    return v_wave

# ============================================================
# STATIC ILLUSTRATION
# ============================================================
def plot_static():
    """Plot disk A, disk B, and their superposition."""
    holes = make_spiral_holes()
    shutters = make_shutters(rotation_angle=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Disk A: spiral aperture array ---
    ax = axes[0]
    ax.set_title("Disk A: spiral with 120 apertures\n(4 turns × 30)")
    circle_outer = plt.Circle((0, 0), D_DISK/2, fill=False, color='black', lw=1)
    ax.add_patch(circle_outer)

    phi_smooth = np.linspace(0, N_TURNS * 2*np.pi, 1000)
    r_smooth = R_INNER + (DR_TURN / (2*np.pi)) * phi_smooth
    ax.plot(r_smooth*np.cos(phi_smooth), r_smooth*np.sin(phi_smooth),
            'b-', alpha=0.3, lw=0.7)

    for (x, y, r, phi, idx) in holes:
        c = plt.Circle((x, y), HOLE_DIAM/2, color='steelblue', alpha=0.8)
        ax.add_patch(c)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")

    # --- Disk B: shutter family ---
    ax = axes[1]
    ax.set_title("Disk B: 30 inclined shutters\n(inclination = 12° over radial span)")
    circle_outer = plt.Circle((0, 0), D_DISK/2, fill=False, color='black', lw=1)
    ax.add_patch(circle_outer)
    for (x_arr, y_arr, a0) in shutters:
        ax.plot(x_arr, y_arr, 'r-', lw=2, alpha=0.7)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")

    # --- Superposition: disk A + disk B + open apertures ---
    ax = axes[2]
    ax.set_title("Superposition: open apertures\nat shutter intersections")
    circle_outer = plt.Circle((0, 0), D_DISK/2, fill=False, color='black', lw=1)
    ax.add_patch(circle_outer)
    ax.plot(r_smooth*np.cos(phi_smooth), r_smooth*np.sin(phi_smooth),
            'b-', alpha=0.2, lw=0.7)

    open_holes = find_open_holes(holes, rotation_angle=0.0, tolerance_rad=np.radians(2))
    open_idx_set = set(h[4] for h in open_holes)
    for (x, y, r, phi, idx) in holes:
        if idx in open_idx_set:
            c = plt.Circle((x, y), HOLE_DIAM/2, color='lime', alpha=1.0,
                           ec='darkgreen', lw=1.5)
        else:
            c = plt.Circle((x, y), HOLE_DIAM/2, color='lightgray', alpha=0.5)
        ax.add_patch(c)

    for (x_arr, y_arr, a0) in shutters:
        ax.plot(x_arr, y_arr, 'r-', lw=1, alpha=0.4)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")

    plt.suptitle("MSPT-D: static geometry of the disk spiral-phased transducer",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("MSPT_D_static.png", dpi=120, bbox_inches='tight')
    plt.show()

# ============================================================
# ROTATION ANIMATION
# ============================================================
def animate_svp():
    """Animate the relative rotation of disk B with respect to disk A."""
    holes = make_spiral_holes()

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-85, 85)
    ax.set_ylim(-85, 85)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")

    # Background spiral.
    phi_smooth = np.linspace(0, N_TURNS * 2*np.pi, 1000)
    r_smooth = R_INNER + (DR_TURN / (2*np.pi)) * phi_smooth
    ax.plot(r_smooth*np.cos(phi_smooth), r_smooth*np.sin(phi_smooth),
            'b-', alpha=0.2, lw=0.7)

    # All apertures are initially shown in grey.
    hole_patches = []
    for (x, y, r, phi, idx) in holes:
        c = plt.Circle((x, y), HOLE_DIAM/2, color='lightgray', alpha=0.5)
        ax.add_patch(c)
        hole_patches.append(c)

    # Shutter curves updated during animation.
    shutter_lines = []
    for _ in range(N_SHUTTERS):
        line, = ax.plot([], [], 'r-', lw=1.5, alpha=0.5)
        shutter_lines.append(line)

    title = ax.set_title("")

    def init():
        return hole_patches + shutter_lines + [title]

    def update(frame):
        # Slow visual animation: one aperture pitch is traversed in 60 frames.
        rotation_angle = frame * (DPHI_HOLE / 60.0)

        shutters = make_shutters(rotation_angle)
        for line, (x_arr, y_arr, a0) in zip(shutter_lines, shutters):
            line.set_data(x_arr, y_arr)

        open_holes = find_open_holes(holes, rotation_angle, tolerance_rad=np.radians(2.5))
        open_idx_set = set(h[4] for h in open_holes)

        for patch, (x, y, r, phi, idx) in zip(hole_patches, holes):
            if idx in open_idx_set:
                patch.set_color('lime')
                patch.set_alpha(1.0)
            else:
                patch.set_color('lightgray')
                patch.set_alpha(0.4)

        title.set_text(
            f"MSPT-D: rotation = {np.degrees(rotation_angle):.2f}° "
            f"(pitch = {np.degrees(DPHI_HOLE):.1f}°) | "
            f"open apertures: {len(open_idx_set)}"
        )
        return hole_patches + shutter_lines + [title]

    # Four complete visual sweep cycles.
    n_frames = 60 * 4
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=50, blit=False, repeat=True)

    plt.tight_layout()

    try:
        anim.save("MSPT_D_animation.gif", writer='pillow', fps=20)
        print("Animation saved as MSPT_D_animation.gif")
    except Exception as e:
        print(f"Could not save GIF animation: {e}")

    plt.show()
    return anim

# ============================================================
# RUNNING COINCIDENCE POINT PLOT
# ============================================================
def plot_running_point():
    """
    Plot the radial coordinate of the open coincidence point versus time.

    During one DPHI_HOLE rotation, the coincidence point is intended
    to sweep from R_INNER to R_OUTER.
    """
    holes = make_spiral_holes()
    n_steps = 300
    angles = np.linspace(0, DPHI_HOLE * 2, n_steps)  # two visual sweeps

    open_radii = []
    open_times = []
    for i, ang in enumerate(angles):
        open_h = find_open_holes(holes, ang, tolerance_rad=np.radians(2.5))
        for h in open_h:
            open_times.append(i / n_steps * 2)  # in units of one sweep
            open_radii.append(h[2])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(open_times, open_radii, c='green', s=15, alpha=0.7)
    ax.set_xlabel("Time, in units of one sweep")
    ax.set_ylabel("Radius of open aperture, mm")
    ax.set_title("Travelling coincidence point: open-aperture radius versus time\n"
                 "(each branch corresponds to one spiral sweep)")
    ax.grid(alpha=0.3)
    ax.axhline(R_INNER, color='gray', ls='--', alpha=0.5, label=f'R_inner = {R_INNER}')
    ax.axhline(R_OUTER, color='gray', ls='--', alpha=0.5, label=f'R_outer = {R_OUTER}')
    ax.legend()
    plt.tight_layout()
    plt.savefig("MSPT_D_running_point.png", dpi=120, bbox_inches='tight')
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    v_wave = print_kinematics()
    print("\n[1/3] Drawing static geometry...")
    plot_static()
    print("\n[2/3] Drawing the travelling coincidence-point diagram...")
    plot_running_point()
    print("\n[3/3] Running the rotation animation...")
    anim = animate_svp()
