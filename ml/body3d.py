"""
ml/body3d.py — Visualisation 3D stylisée du corps + chaînes musculaires.

Chaque zone anatomique est représentée par un ellipsoïde dont la couleur
varie du gris-bleu (sain) au rouge vif (zone à risque). Les chaînes
musculaires sont tracées comme des tubes cylindriques reliant les zones.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Géométrie : (centre_x, centre_y, centre_z, ray_x, ray_y, ray_z)
BODY_GEOMETRY: Dict[str, tuple] = {
    "head":            (0.0,  0.0,  9.0, 0.9, 0.9, 1.1),
    "chest":           (0.0,  0.0,  6.5, 2.2, 1.3, 1.7),
    "heart":           (-0.4,-0.9,  6.6, 0.5, 0.4, 0.6),
    "abdomen":         (0.0,  0.0,  4.6, 1.7, 1.1, 1.2),
    "lower_back":      (0.0,  0.9,  4.6, 1.6, 0.5, 1.0),
    "shoulders":       (0.0,  0.0,  7.8, 2.8, 1.0, 0.5),
    "left_arm":        (-2.7, 0.0,  5.7, 0.5, 0.5, 1.8),
    "right_arm":       ( 2.7, 0.0,  5.7, 0.5, 0.5, 1.8),
    "left_quad":       (-0.9, 0.0,  2.6, 0.7, 0.8, 1.4),
    "right_quad":      ( 0.9, 0.0,  2.6, 0.7, 0.8, 1.4),
    "left_hamstring":  (-0.9, 0.6,  2.6, 0.7, 0.4, 1.3),
    "right_hamstring": ( 0.9, 0.6,  2.6, 0.7, 0.4, 1.3),
    "left_knee":       (-0.9, 0.0,  1.0, 0.55,0.55,0.35),
    "right_knee":      ( 0.9, 0.0,  1.0, 0.55,0.55,0.35),
    "left_calf":       (-0.9, 0.2, -0.4, 0.55,0.6, 1.0),
    "right_calf":      ( 0.9, 0.2, -0.4, 0.55,0.6, 1.0),
    "left_ankle":      (-0.9, 0.0, -1.7, 0.45,0.5, 0.3),
    "right_ankle":     ( 0.9, 0.0, -1.7, 0.45,0.5, 0.3),
}

# Palette risk : vert mint → bleu → ambre → rouge → rouge foncé
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", ["#22D3A5", "#3B82F6", "#F59E0B", "#EF4444", "#B91C1C"])

# ── Muscle chains ─────────────────────────────────────────────────────────────
# Each chain: (name_fr, colour, [body_part_keys ordered along the chain])
MUSCLE_CHAINS = [
    # Posterior chain (hamstrings → lower back → spine)
    ("Chaîne post.", "#EF4444",
     ["left_ankle","left_calf","left_hamstring","lower_back",
      "right_hamstring","right_calf","right_ankle"]),
    # Anterior chain (quads → abdomen → chest)
    ("Chaîne ant.", "#3B82F6",
     ["left_ankle","left_knee","left_quad","abdomen","chest",
      "right_quad","right_knee","right_ankle"]),
    # Lateral / shoulder chain
    ("Chaîne lat.", "#F59E0B",
     ["left_arm","shoulders","right_arm"]),
    # Deep core chain (lower_back → abdomen → chest → head)
    ("Chaîne core", "#22D3A5",
     ["lower_back","abdomen","chest","head"]),
]

_MESH_N = 12

def _ellipsoid(cx, cy, cz, rx, ry, rz, n=_MESH_N):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    sv = np.sin(v)
    x = cx + rx * np.outer(np.cos(u), sv)
    y = cy + ry * np.outer(np.sin(u), sv)
    z = cz + rz * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z

_MESH_CACHE: Dict[str, tuple] = {
    part: _ellipsoid(*geom) for part, geom in BODY_GEOMETRY.items()
}

def _chain_color_along(chain_color: str, risk_a: float, risk_b: float) -> str:
    """Blend the chain's base colour toward red proportionally to segment risk."""
    avg_risk = (risk_a + risk_b) / 2.0
    if avg_risk < 0.25:
        return chain_color
    # interpolate toward danger red
    import matplotlib.colors as mc
    base = np.array(mc.to_rgb(chain_color))
    red  = np.array(mc.to_rgb("#EF4444"))
    t = min(avg_risk / 0.8, 1.0)
    mixed = base * (1-t) + red * t
    return f"#{int(mixed[0]*255):02x}{int(mixed[1]*255):02x}{int(mixed[2]*255):02x}"

def _tube_between(ax, p0, p1, radius=0.12, color="#888888", alpha=0.7, n=8):
    """Draw a cylindrical tube between two 3D points."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return
    axis = vec / length

    # Build orthonormal frame
    perp = np.array([1,0,0]) if abs(axis[0]) < 0.9 else np.array([0,1,0])
    u_hat = np.cross(axis, perp); u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(axis, u_hat)

    t_vals = np.linspace(0, 1, 6)
    theta  = np.linspace(0, 2*np.pi, n)

    for t0, t1 in zip(t_vals[:-1], t_vals[1:]):
        for th0, th1 in zip(theta[:-1], theta[1:]):
            corners = []
            for t, th in [(t0,th0),(t0,th1),(t1,th1),(t1,th0)]:
                pt = p0 + t*vec + radius*(np.cos(th)*u_hat + np.sin(th)*v_hat)
                corners.append(pt)
            xs = [c[0] for c in corners] + [corners[0][0]]
            ys = [c[1] for c in corners] + [corners[0][1]]
            zs = [c[2] for c in corners] + [corners[0][2]]
            ax.plot_surface(
                np.array([[xs[0],xs[1]],[xs[3],xs[2]]]),
                np.array([[ys[0],ys[1]],[ys[3],ys[2]]]),
                np.array([[zs[0],zs[1]],[zs[3],zs[2]]]),
                color=color, linewidth=0, shade=False,
                antialiased=False, alpha=alpha,
            )

def _get_center(part: str):
    g = BODY_GEOMETRY[part]
    return (g[0], g[1], g[2])


def render_body(ax, body_risk: Dict[str, float], title: str = "",
                reset_view: bool = False,
                show_chains: bool = True) -> None:
    """
    Draw the body on an existing matplotlib 3D axis.
    If show_chains=True, also renders muscle chain tubes coloured by risk.
    """
    prev_elev = getattr(ax, "elev", None)
    prev_azim = getattr(ax, "azim", None)
    try:
        prev_xlim = ax.get_xlim3d()
        prev_ylim = ax.get_ylim3d()
        prev_zlim = ax.get_zlim3d()
    except Exception:
        prev_xlim = prev_ylim = prev_zlim = None

    ax.clear()
    ax.set_facecolor("#0B0F14")

    # ── 1. Ellipsoid body segments ─────────────────────────────────────────
    for part, (x, y, z) in _MESH_CACHE.items():
        risk  = float(body_risk.get(part, 0.0))
        color = RISK_CMAP(risk)
        ax.plot_surface(
            x, y, z,
            color=color, linewidth=0,
            antialiased=False, shade=False,
            rstride=2, cstride=2,
            alpha=0.92 if risk > 0.2 else 0.65,
        )

    # ── 2. Muscle chains ──────────────────────────────────────────────────
    if show_chains:
        for chain_name, chain_color, parts in MUSCLE_CHAINS:
            valid = [p for p in parts if p in BODY_GEOMETRY]
            for i in range(len(valid) - 1):
                pa, pb = valid[i], valid[i+1]
                risk_a = float(body_risk.get(pa, 0.0))
                risk_b = float(body_risk.get(pb, 0.0))
                seg_color = _chain_color_along(chain_color, risk_a, risk_b)
                seg_risk   = (risk_a + risk_b) / 2.0
                alpha = 0.55 + 0.4 * seg_risk  # more opaque when risky
                _tube_between(ax, _get_center(pa), _get_center(pb),
                               radius=0.14, color=seg_color, alpha=alpha)

    # ── 3. Legend for chains ───────────────────────────────────────────────
    if show_chains:
        from matplotlib.lines import Line2D
        handles = [Line2D([0],[0], color=c, lw=3, label=n)
                   for n, c, _ in MUSCLE_CHAINS]
        ax.legend(handles=handles, loc="upper left", fontsize=7,
                  framealpha=0.3, labelcolor="white",
                  facecolor="#1E2A42", edgecolor="#243042")

    ax.set_box_aspect((1, 0.85, 1.8))
    ax.set_axis_off()

    first_render = prev_xlim is None or prev_xlim == (0.0, 1.0)
    if reset_view or first_render:
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.0, 3.0)
        ax.set_zlim(-2.5, 10.5)
        ax.view_init(elev=12, azim=-72)
    else:
        ax.set_xlim3d(prev_xlim)
        ax.set_ylim3d(prev_ylim)
        ax.set_zlim3d(prev_zlim)
        ax.view_init(elev=prev_elev, azim=prev_azim)

    if not getattr(ax, "_mouse_inited", False):
        try:
            ax.mouse_init()
            ax._mouse_inited = True
        except Exception:
            pass

    if title:
        ax.set_title(title, color="#E6EDF3", fontsize=11,
                     fontweight="bold", pad=6)
