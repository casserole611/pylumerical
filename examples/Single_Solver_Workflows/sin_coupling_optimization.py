# SiN Coupling Optimisation (MODE FDE)
#
# Maximises power coupling from a pre-computed reference mode ("global_mode1")
# into a SiN waveguide rectangle whose thickness is fixed at 200 nm.
#
# The script performs a two-stage search:
#   1. Coarse sweep across a configurable width range.
#   2. Bounded golden-section refinement around the best coarse point.
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
SIN_THICKNESS   = 200e-9      # Fixed SiN thickness (m) – used only for sanity checks
RECT_NAME       = "rectangle" # Name of the SiN rectangle in the deck
GLOBAL_MODE     = "global_mode1"  # Name of the reference mode dataset

# Width sweep range
WIDTH_MIN   = 10e-9    # m  – minimum fabricable feature size
WIDTH_MAX   = 10e-6    # m  – upper bound (peak coupling expected well below this)
N_COARSE    = 30       # Number of coarse sweep points (log-spaced, see below)

# Multi-waveguide gap constraint (enforced when extending to multi-rect designs)
MIN_GAP     = 10e-9    # m  – minimum edge-to-edge gap between SiN rectangles

# FDE analysis settings
N_TRIAL_MODES = 20

# ── Core helpers ──────────────────────────────────────────────────────────────

def _set_width(session, width: float) -> None:
    """Update the SiN rectangle width without touching other dimensions."""
    session.setnamed(RECT_NAME, "x span", width)


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


def _coupling_to_global(session, n_modes: int) -> float:
    """
    Return the maximum power coupling coefficient between global_mode1 and
    any mode found by the current FDE run.

    Lumerical's overlap() function computes the bidirectional overlap integral:
        η = |∫∫(E₁×H₂* + E₂×H₁*)·ẑ dA|² /
            (4 · Re{∫∫(E₁×H₁*)·ẑ dA} · Re{∫∫(E₂×H₂*)·ẑ dA})
    which equals the fraction of input power that couples into the target mode.
    """
    script = f"""
    _max_c = 0;
    for (_i = 1; _i <= {n_modes}; _i = _i + 1) {{
        _c = overlap("FDE::data::{GLOBAL_MODE}", "FDE::data::mode" + num2str(_i));
        if (_c > _max_c) {{ _max_c = _c; }}
    }}
    """
    session.eval(script)
    return float(session.getv("_max_c"))


def compute_coupling(session, width: float) -> float:
    """Set width, run FDE, return best coupling to global_mode1."""
    _set_width(session, width)
    n = _find_modes(session)
    return _coupling_to_global(session, n)


# ── Optimisation ─────────────────────────────────────────────────────────────

def optimise(session) -> tuple[float, float, list, list]:
    """
    Two-stage width optimisation.

    Returns
    -------
    best_width   : optimal SiN width (m)
    best_coupling: corresponding power coupling coefficient
    widths       : coarse-sweep width array
    couplings    : coarse-sweep coupling array
    """
    # Log-spaced so the 10 nm–10 µm range is sampled evenly per decade
    widths    = np.logspace(np.log10(WIDTH_MIN), np.log10(WIDTH_MAX), N_COARSE)
    couplings = np.empty(N_COARSE)

    print(f"\n{'Width (nm)':>12}  {'Coupling':>10}")
    print("─" * 26)
    for idx, w in enumerate(widths):
        c = compute_coupling(session, w)
        couplings[idx] = c
        print(f"{w * 1e9:>12.1f}  {c:>10.6f}")

    # Bracket around the coarse maximum
    best_idx = int(np.argmax(couplings))
    w_lo = widths[max(0, best_idx - 1)]
    w_hi = widths[min(N_COARSE - 1, best_idx + 1)]

    print(f"\nFine search:  [{w_lo * 1e9:.1f} nm, {w_hi * 1e9:.1f} nm]")
    result = minimize_scalar(
        lambda w: -compute_coupling(session, w),
        bounds=(w_lo, w_hi),
        method="bounded",
        options={"xatol": 1e-9, "maxiter": 60},
    )

    best_width    = float(result.x)
    best_coupling = float(-result.fun)

    print(f"\n{'─'*40}")
    print(f"  Optimal SiN width  : {best_width * 1e9:.2f} nm")
    print(f"  Max power coupling : {best_coupling:.6f}  ({best_coupling * 100:.2f} %)")
    print(f"{'─'*40}\n")

    return best_width, best_coupling, widths.tolist(), couplings.tolist()


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_sweep(widths, couplings, best_width, best_coupling) -> None:
    """Plot coarse sweep with the optimal point highlighted."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.array(widths) * 1e9, couplings, "o-", lw=1.5, label="Coarse sweep")
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
