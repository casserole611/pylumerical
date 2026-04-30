# SiN Coupling Optimisation (MODE FDE)
#
# Maximises power coupling from a pre-computed reference mode ("global_mode1")
# into a SiN waveguide rectangle whose thickness is fixed at 200 nm.
#
# The script performs a three-stage search:
#   1. Coarse log-spaced sweep across the full width range.
#   2. Adaptive refinement: intervals where coupling changes steeply are
#      recursively bisected (log-midpoint) until smooth or below min feature size.
#   3. Bounded golden-section refinement around the global maximum.
#
# Prerequisites:
#   - Valid MODE licence.
#   - An open/loaded .lms file that contains:
#       * A rectangle named "rectangle" (SiN, y-thickness = 200 nm).
#       * A mode dataset named "global_mode1" (FDE::data::global_mode1).
#       * An FDE solver region already configured for the cross-section.

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import ansys.lumerical.core as lumapi

# ── Configurable parameters ───────────────────────────────────────────────────

SIMULATION_FILE = ""          # Path to .lms file; leave "" to use current session
WAVELENGTH      = 780e-9      # Wavelength (m)
SIN_THICKNESS   = 200e-9      # Fixed z span of the SiN rectangle (m) – not modified
RECT_NAME       = "rectangle" # Name of the SiN rectangle in the deck
GLOBAL_MODE     = "global_mode1"  # Name of the reference mode dataset

# Pre-configured FDE region (not modified by this script – geometry is fixed in the deck)
#   z span : set to cover the SiO2 cladding/substrate regions
#   y span : 10 µm  ← hard upper bound for the waveguide width sweep below
FDE_Y_SPAN  = 10e-6    # m  – FDE window y span; WIDTH_MAX must stay below this

# y span sweep range (x = propagation direction, z = thickness – both untouched)
WIDTH_MIN   = 10e-9    # m  – minimum fabricable feature size
WIDTH_MAX   = FDE_Y_SPAN   # m  – sweeps up to the FDE window edge; peak expected well below
N_COARSE    = 30       # Number of coarse sweep points (log-spaced)

# Multi-waveguide gap constraint (enforced when extending to multi-rect designs)
MIN_GAP     = 10e-9    # m  – minimum edge-to-edge gap between SiN rectangles

# Adaptive refinement: after the coarse sweep, intervals where coupling changes
# by more than ADAPT_REL_TOL * (max−min) are subdivided at their log-midpoint,
# recursively, until the interval narrows to ADAPT_MIN_WIDTH or ADAPT_MAX_DEPTH
# recursion levels are reached.
ADAPT_REL_TOL  = 0.02   # 2 % of the observed coupling range
ADAPT_MIN_WIDTH = 1e-9  # m  – stop subdividing below this interval width
ADAPT_MAX_DEPTH = 6     # maximum recursion depth per interval

# FDE analysis settings – find 5 modes for the SiN waveguide, use only TE0
N_TRIAL_MODES = 5

# ── Core helpers ──────────────────────────────────────────────────────────────

def _set_width(session, width: float) -> None:
    """Update the SiN rectangle y span (width); x=propagation, z=thickness are untouched."""
    session.setnamed(RECT_NAME, "y span", width)


def _find_modes(session) -> int:
    """Run FDE eigenmode search; return number of modes found."""
    session.setanalysis("wavelength", WAVELENGTH)
    session.setanalysis("number of trial modes", N_TRIAL_MODES)
    session.setanalysis("search", "near n")
    session.setanalysis("use max index", True)
    session.findmodes()

    # Retrieve how many modes were found via a short Lumerical script
    session.eval("_n_modes = findstring(getresult('FDE::data', 'list'), 'mode');")
    try:
        n = int(session.getv("_n_modes"))
    except Exception:
        n = 1
    return n


def _find_te0(session, n_modes: int) -> int:
    """
    Return the 1-based index of TE0: the highest-neff mode with TE fraction > 0.5.
    FDE sorts modes by decreasing neff, so the first TE mode encountered is TE0.
    Returns 0 if no TE mode is found (waveguide below cut-off for TE).
    """
    script = f"""
    _te0_idx = 0;
    for (_i = 1; _i <= {n_modes}; _i = _i + 1) {{
        if (_te0_idx == 0) {{
            _mname = "FDE::data::mode" + num2str(_i);
            _te = getresult(_mname, "TE polarization fraction");
            if (_te > 0.5) {{ _te0_idx = _i; }}
        }}
    }}
    """
    session.eval(script)
    return int(session.getv("_te0_idx"))


def _coupling_to_global(session, te0_idx: int) -> float:
    """
    Power coupling coefficient from global_mode1 into the SiN TE0 mode.

    Uses copydcard to avoid mutating the original dataset, then shiftdcard to
    align the global mode's y-centre with the SiN rectangle centre before
    evaluating the bidirectional overlap integral:
        η = |∫∫(E₁×H₂* + E₂×H₁*)·ẑ dA|² /
            (4 · Re{∫∫(E₁×H₁*)·ẑ dA} · Re{∫∫(E₂×H₂*)·ẑ dA})
    """
    script = f"""
    copydcard("FDE::data::{GLOBAL_MODE}", "working_global_mode");
    _y_sin    = getnamed("{RECT_NAME}", "y");
    _y_global = mean(getdata("working_global_mode", "y"));
    shiftdcard("working_global_mode", 0, _y_sin - _y_global);
    _coupling = overlap("working_global_mode", "FDE::data::mode{te0_idx}");
    """
    session.eval(script)
    return float(session.getv("_coupling"))


def compute_coupling(session, width: float) -> float:
    """Set y span, find 5 modes, pick TE0, return aligned overlap with global_mode1."""
    _set_width(session, width)
    n       = _find_modes(session)
    te0_idx = _find_te0(session, n)
    if te0_idx == 0:
        return 0.0  # no TE mode supported at this width
    return _coupling_to_global(session, te0_idx)


# ── Adaptive refinement ───────────────────────────────────────────────────────

def _refine_interval(
    session,
    w_lo: float, c_lo: float,
    w_hi: float, c_hi: float,
    depth: int,
    coupling_range: float,
) -> list[tuple[float, float]]:
    """
    Recursively insert (width, coupling) samples inside [w_lo, w_hi].

    Returns a list of NEW interior points (not including the endpoints).
    Subdivision stops when:
      - the coupling change across the interval is within tolerance, OR
      - the interval is narrower than ADAPT_MIN_WIDTH, OR
      - the recursion depth exceeds ADAPT_MAX_DEPTH.
    """
    if depth >= ADAPT_MAX_DEPTH:
        return []
    if (w_hi - w_lo) <= ADAPT_MIN_WIDTH * 2:
        return []
    if abs(c_hi - c_lo) <= ADAPT_REL_TOL * coupling_range:
        return []

    w_mid = np.exp((np.log(w_lo) + np.log(w_hi)) / 2)
    c_mid = compute_coupling(session, w_mid)
    print(f"  [refine d={depth}] {w_mid * 1e9:9.2f} nm  →  {c_mid:.6f}")

    left  = _refine_interval(session, w_lo, c_lo, w_mid, c_mid, depth + 1, coupling_range)
    right = _refine_interval(session, w_mid, c_mid, w_hi, c_hi, depth + 1, coupling_range)
    return left + [(w_mid, c_mid)] + right


def _adaptive_refine(
    session,
    pts: list[tuple[float, float]],
    coupling_range: float,
) -> list[tuple[float, float]]:
    """
    Given a sorted list of (width, coupling) pairs, return a refined list
    with extra samples inserted wherever the coupling landscape is steep.
    """
    refined = [pts[0]]
    for i in range(len(pts) - 1):
        interior = _refine_interval(
            session,
            pts[i][0], pts[i][1],
            pts[i + 1][0], pts[i + 1][1],
            depth=0,
            coupling_range=coupling_range,
        )
        refined.extend(interior)
        refined.append(pts[i + 1])
    return refined


# ── Optimisation ─────────────────────────────────────────────────────────────

def optimise(session) -> tuple[float, float, list, list]:
    """
    Two-stage width optimisation.

    Returns
    -------
    best_width   : optimal SiN width (m)
    best_coupling: corresponding power coupling coefficient
    widths       : all sampled widths (coarse + adaptive), sorted
    couplings    : corresponding coupling values
    """
    # ── Stage 1: coarse log-spaced sweep ─────────────────────────────────────
    coarse_w = np.logspace(np.log10(WIDTH_MIN), np.log10(WIDTH_MAX), N_COARSE)

    print(f"\n── Coarse sweep ({N_COARSE} points, log-spaced) ──")
    print(f"{'Width (nm)':>12}  {'Coupling':>10}")
    print("─" * 26)
    pts: list[tuple[float, float]] = []
    for w in coarse_w:
        c = compute_coupling(session, w)
        pts.append((w, c))
        print(f"{w * 1e9:>12.1f}  {c:>10.6f}")

    # ── Stage 2: adaptive refinement ─────────────────────────────────────────
    c_vals = [p[1] for p in pts]
    coupling_range = max(c_vals) - min(c_vals)

    if coupling_range > 0:
        print(
            f"\n── Adaptive refinement "
            f"(tol = {ADAPT_REL_TOL*100:.0f}% × range = {coupling_range * ADAPT_REL_TOL:.6f}) ──"
        )
        pts = _adaptive_refine(session, pts, coupling_range)
    else:
        print("\n  Coupling is flat across the sweep range – skipping adaptive refinement.")

    widths    = [p[0] for p in pts]
    couplings = [p[1] for p in pts]

    # ── Stage 3: golden-section refinement around the global maximum ──────────
    best_idx = int(np.argmax(couplings))
    w_lo = widths[max(0, best_idx - 1)]
    w_hi = widths[min(len(widths) - 1, best_idx + 1)]

    print(f"\n── Fine search: [{w_lo * 1e9:.2f} nm, {w_hi * 1e9:.2f} nm] ──")
    result = minimize_scalar(
        lambda w: -compute_coupling(session, w),
        bounds=(w_lo, w_hi),
        method="bounded",
        options={"xatol": 1e-9, "maxiter": 60},
    )

    best_width    = float(result.x)
    best_coupling = float(-result.fun)

    print(f"\n{'─'*42}")
    print(f"  Optimal SiN width  : {best_width * 1e9:.2f} nm")
    print(f"  Max power coupling : {best_coupling:.6f}  ({best_coupling * 100:.2f} %)")
    print(f"  Total FDE runs     : {len(pts)}")
    print(f"{'─'*42}\n")

    return best_width, best_coupling, widths, couplings


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_sweep(widths, couplings, best_width, best_coupling) -> None:
    """Plot coarse sweep with the optimal point highlighted."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.array(widths) * 1e9, couplings, "o-", lw=1.5, label="All samples (coarse + adaptive)")
    ax.axvline(best_width * 1e9, color="red", ls="--", lw=1.2, label=f"Optimal: {best_width*1e9:.1f} nm")
    ax.scatter([best_width * 1e9], [best_coupling], color="red", zorder=5)
    ax.set_xlabel("SiN width (nm)")
    ax.set_ylabel("Power coupling coefficient")
    ax.set_title("Coupling from global_mode1 into SiN waveguide vs. width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_optimal_mode(session) -> None:
    """Visualise the E-field of the best-coupled mode at the optimal width."""
    # Mode 1 after the final findmodes call is the first (usually best) mode
    Efield = session.getresult("FDE::data::mode1", "E")
    x, y   = Efield["x"], Efield["y"]
    E      = Efield["E"]
    E_mag  = np.abs(E[:, :, 0, 0, 0])**2 + np.abs(E[:, :, 0, 0, 1])**2 + np.abs(E[:, :, 0, 0, 2])**2
    X, Y   = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(5, 5))
    cf = ax.contourf(X * 1e9, Y * 1e9, np.transpose(E_mag), levels=40, cmap="inferno")
    plt.colorbar(cf, ax=ax, label="|E|²  (a.u.)")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("E-field magnitude at optimal SiN width")
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    session = lumapi.MODE(hide=False)

    if SIMULATION_FILE:
        session.load(SIMULATION_FILE)

    best_width, best_coupling, widths, couplings = optimise(session)

    # Apply optimal width and run once more for final inspection
    compute_coupling(session, best_width)

    plot_sweep(widths, couplings, best_width, best_coupling)
    plot_optimal_mode(session)


if __name__ == "__main__":
    main()
