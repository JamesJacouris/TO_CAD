"""
Generate a TikZ/LaTeX figure showing the Wall_Bracket TO problem geometry and BCs.

Outputs a standalone .tex file that can be compiled directly with pdflatex,
or the tikzpicture environment can be pasted into a report.

Usage:
    python scripts/fig_problem_setup.py
    pdflatex output/hybrid_v2/Wall_Bracket_problem_setup.tex
"""
import numpy as np

# ── Load NPZ to get actual BC geometry ────────────────────────────────────────
npz_path = "output/hybrid_v2/Wall_Bracket_top3d.npz"
data = np.load(npz_path, allow_pickle=True)
bc_tags = data['bc_tags']  # shape (nely, nelx, nelz) = (30, 60, 30)
pitch = float(data['pitch']) if 'pitch' in data else 1.0

nely, nelx, nelz = bc_tags.shape

# Find BC locations in world coords
fixed_ijk = np.argwhere(bc_tags == 1)
loaded_ijk = np.argwhere(bc_tags == 2)
fixed_xyz = fixed_ijk[:, [1, 0, 2]] * pitch + pitch * 0.5
loaded_xyz = loaded_ijk[:, [1, 0, 2]] * pitch + pitch * 0.5
load_centroid = loaded_xyz.mean(axis=0)

# Fixed region bounds
fix_y_range = (fixed_xyz[:, 1].min() - 0.5, fixed_xyz[:, 1].max() + 0.5)
fix_z_range = (fixed_xyz[:, 2].min() - 0.5, fixed_xyz[:, 2].max() + 0.5)

# Load region bounds
load_y_range = (loaded_xyz[:, 1].min() - 0.5, loaded_xyz[:, 1].max() + 0.5)
load_z_range = (loaded_xyz[:, 2].min() - 0.5, loaded_xyz[:, 2].max() + 0.5)

print(f"Domain: {nelx} x {nely} x {nelz}")
print(f"Fixed face: Y=[{fix_y_range[0]:.0f},{fix_y_range[1]:.0f}], "
      f"Z=[{fix_z_range[0]:.0f},{fix_z_range[1]:.0f}] at X=0")
print(f"Load region: Y=[{load_y_range[0]:.0f},{load_y_range[1]:.0f}], "
      f"Z=[{load_z_range[0]:.0f},{load_z_range[1]:.0f}] at X={nelx}")
print(f"Load centroid: {load_centroid}")

# ── TikZ scaling ──────────────────────────────────────────────────────────────
# Scale to fit nicely on page (domain is 60x30x30, scale to ~8cm wide)
s = 0.12  # scale factor: 60 * 0.12 = 7.2cm

# Isometric-ish projection angles
alpha = 30  # degrees — X-axis tilt
beta = 15   # degrees — depth foreshortening

# 3D to 2D projection (cabinet oblique — simpler and clearer for engineering)
# X → right, Y → up, Z → diagonal (foreshortened)
import math
ca, sa = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
depth_scale = 0.5  # foreshorten depth axis

def proj(x, y, z):
    """Project 3D (x,y,z) to 2D (u,v) using oblique projection."""
    u = x * s + z * s * depth_scale * ca
    v = y * s + z * s * depth_scale * sa
    return u, v

# ── Generate TikZ ────────────────────────────────────────────────────────────
Lx, Ly, Lz = float(nelx), float(nely), float(nelz)

tex = r"""\documentclass[border=5mm]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, patterns, decorations.pathreplacing, calc}

\begin{document}
\begin{tikzpicture}[
    >=Stealth,
    thick,
    support/.style={blue!70!black, line width=1pt},
    load/.style={red!80!black, line width=1.5pt, -{Stealth[length=4mm, width=3mm]}},
    dim/.style={gray, thin, |<->|, >=Stealth},
    dimlabel/.style={font=\footnotesize, fill=white, inner sep=1pt},
]

"""

# Helper to format coordinates
def pt(x, y, z):
    u, v = proj(x, y, z)
    return f"({u:.3f},{v:.3f})"

# ── Draw back edges (dashed) ─────────────────────────────────────────────────
tex += "    % Back edges (hidden)\n"
tex += "    \\draw[gray!40, dashed, thin]\n"
tex += f"        {pt(0, Ly, 0)} -- {pt(Lx, Ly, 0)}\n"
tex += f"        {pt(Lx, Ly, 0)} -- {pt(Lx, 0, 0)}\n"
tex += f"        {pt(Lx, Ly, 0)} -- {pt(Lx, Ly, Lz)};\n\n"

# ── Draw domain box (visible edges) ──────────────────────────────────────────
tex += "    % Design domain (visible edges)\n"
tex += "    \\draw[black, line width=0.8pt]\n"
# Bottom face visible edges
tex += f"        {pt(0, 0, 0)} -- {pt(Lx, 0, 0)}\n"
tex += f"        {pt(0, 0, 0)} -- {pt(0, Ly, 0)}\n"
tex += f"        {pt(0, 0, 0)} -- {pt(0, 0, Lz)}\n"
# Top face visible edges
tex += f"        {pt(0, 0, Lz)} -- {pt(Lx, 0, Lz)}\n"
tex += f"        {pt(0, 0, Lz)} -- {pt(0, Ly, Lz)}\n"
# Right face visible edges
tex += f"        {pt(Lx, 0, 0)} -- {pt(Lx, 0, Lz)}\n"
# Top-back edges
tex += f"        {pt(0, Ly, Lz)} -- {pt(Lx, Ly, Lz)}\n"
tex += f"        {pt(Lx, 0, Lz)} -- {pt(Lx, Ly, Lz)}\n"
tex += f"        {pt(0, Ly, 0)} -- {pt(0, Ly, Lz)}\n"
tex += f"        {pt(Lx, 0, 0)} -- {pt(Lx, 0, 0)};\n\n"  # redundant, end path

# ── Fixed support face (x=0, blue shaded) ────────────────────────────────────
fy0, fy1 = fix_y_range[0], fix_y_range[1]
fz0, fz1 = fix_z_range[0], fix_z_range[1]

tex += "    % Fixed support face\n"
tex += f"    \\fill[blue!20, opacity=0.6]\n"
tex += f"        {pt(0, fy0, fz0)} -- {pt(0, fy1, fz0)} -- {pt(0, fy1, fz1)} -- {pt(0, fy0, fz1)} -- cycle;\n"
tex += f"    \\draw[support]\n"
tex += f"        {pt(0, fy0, fz0)} -- {pt(0, fy1, fz0)} -- {pt(0, fy1, fz1)} -- {pt(0, fy0, fz1)} -- cycle;\n\n"

# Ground hatch marks (diagonal lines behind the support face)
tex += "    % Ground hatching\n"
n_hatch = 6
hatch_len = 1.5  # length of hatch marks in mm
for i in range(n_hatch + 1):
    t = i / n_hatch
    # Along Y at fixed Z positions
    y_h = fy0 + t * (fy1 - fy0)
    u1, v1 = proj(-hatch_len, y_h, fz0)
    u2, v2 = proj(0, y_h, fz0)
    tex += f"    \\draw[support, thin] ({u1:.3f},{v1:.3f}) -- ({u2:.3f},{v2:.3f});\n"
# Along Z at bottom
for i in range(n_hatch + 1):
    t = i / n_hatch
    z_h = fz0 + t * (fz1 - fz0)
    u1, v1 = proj(-hatch_len, fy0, z_h)
    u2, v2 = proj(0, fy0, z_h)
    tex += f"    \\draw[support, thin] ({u1:.3f},{v1:.3f}) -- ({u2:.3f},{v2:.3f});\n"

tex += "\n"

# ── Load region & arrow ──────────────────────────────────────────────────────
ly0, ly1 = load_y_range
lz0, lz1 = load_z_range
lx = Lx

tex += "    % Load application region\n"
tex += f"    \\fill[red!20, opacity=0.6]\n"
tex += f"        {pt(lx, ly0, lz0)} -- {pt(lx, ly1, lz0)} -- {pt(lx, ly1, lz1)} -- {pt(lx, ly0, lz1)} -- cycle;\n"
tex += f"    \\draw[red!70!black, line width=1pt]\n"
tex += f"        {pt(lx, ly0, lz0)} -- {pt(lx, ly1, lz0)} -- {pt(lx, ly1, lz1)} -- {pt(lx, ly0, lz1)} -- cycle;\n\n"

# Load arrow (pointing -Y from above the load centroid)
lc = load_centroid
arrow_start_y = lc[1] + 12
tex += "    % Load arrow\n"
tex += f"    \\draw[load] {pt(lc[0], arrow_start_y, lc[2])} -- {pt(lc[0], lc[1] + 1, lc[2])};\n\n"

# ── Dimension annotations ────────────────────────────────────────────────────
# X dimension (bottom, front edge)
dim_offset_y = -5
tex += "    % Dimension: X\n"
u0, v0 = proj(0, dim_offset_y, 0)
u1, v1 = proj(Lx, dim_offset_y, 0)
um, vm = proj(Lx/2, dim_offset_y, 0)
tex += f"    \\draw[dim] ({u0:.3f},{v0:.3f}) -- ({u1:.3f},{v1:.3f})\n"
tex += f"        node[midway, below, dimlabel] {{{nelx} elements}};\n\n"

# Y dimension (left side)
dim_offset_x = -5
tex += "    % Dimension: Y\n"
u0, v0 = proj(dim_offset_x, 0, 0)
u1, v1 = proj(dim_offset_x, Ly, 0)
um, vm = proj(dim_offset_x, Ly/2, 0)
tex += f"    \\draw[dim] ({u0:.3f},{v0:.3f}) -- ({u1:.3f},{v1:.3f})\n"
tex += f"        node[midway, left, dimlabel] {{{nely}}};\n\n"

# Z dimension (right side, front)
dim_offset_x2 = Lx + 4
tex += "    % Dimension: Z\n"
u0, v0 = proj(dim_offset_x2, 0, 0)
u1, v1 = proj(dim_offset_x2, 0, Lz)
um, vm = proj(dim_offset_x2, 0, Lz/2)
tex += f"    \\draw[dim] ({u0:.3f},{v0:.3f}) -- ({u1:.3f},{v1:.3f})\n"
tex += f"        node[midway, right, dimlabel] {{{nelz}}};\n\n"

# ── Text labels ──────────────────────────────────────────────────────────────
# Fixed label
u_fix, v_fix = proj(-4, Ly/2, Lz * 0.6)
tex += f"    \\node[blue!70!black, font=\\footnotesize\\bfseries, align=center, anchor=east] at ({u_fix:.3f},{v_fix:.3f})\n"
tex += f"        {{Fixed\\\\(all DOFs)}};\n\n"

# Load label
u_load, v_load = proj(lc[0] + 4, arrow_start_y + 2, lc[2])
tex += f"    \\node[red!80!black, font=\\footnotesize\\bfseries, anchor=west] at ({u_load:.3f},{v_load:.3f})\n"
tex += r"        {$\mathbf{F} = [0,\, {-}10,\, 0]$ N};" + "\n\n"

# Axis labels
u_x, v_x = proj(Lx + 2, -3, 0)
u_y, v_y = proj(-3, Ly + 2, 0)
u_z, v_z = proj(-2, -2, Lz + 3)
tex += f"    \\node[font=\\footnotesize] at ({u_x:.3f},{v_x:.3f}) {{$x$}};\n"
tex += f"    \\node[font=\\footnotesize] at ({u_y:.3f},{v_y:.3f}) {{$y$}};\n"
tex += f"    \\node[font=\\footnotesize] at ({u_z:.3f},{v_z:.3f}) {{$z$}};\n"

tex += r"""
\end{tikzpicture}
\end{document}
"""

# ── Write output ─────────────────────────────────────────────────────────────
out_path = "output/hybrid_v2/Wall_Bracket_problem_setup.tex"
with open(out_path, 'w') as f:
    f.write(tex)
print(f"\nSaved: {out_path}")
print("Compile with: pdflatex output/hybrid_v2/Wall_Bracket_problem_setup.tex")
