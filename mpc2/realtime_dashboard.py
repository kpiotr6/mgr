"""
Real-Time Interactive Cement Mill Dashboard
=============================================
Live, scrolling strip-chart view of the cement grinding circuit simulator
(cement_mill_sim.py), with two sliders you can drag while the simulation
is running:

  * Fresh Feed  (M_C, t/h)       - the raw clinker feed rate into the circuit
  * Separator Speed (rotor rpm)  - controls a *dynamic* (rotor-speed)
                                    separator: higher rpm -> more centrifugal
                                    force -> finer cut point, more
                                    recirculation, finer product; lower rpm
                                    -> coarser cut, less recirculation.
                                    (This is the opposite control sense from
                                    the register-vane control on older static
                                    separators - see cement_mill_sim.py's
                                    set_rotor_speed() for the physics.)

Charts shown, updated every animation frame:
  1. Compartment 1 hold-up  H1 (t)
  2. Compartment 2 hold-up  H2 (t)
  3. Return (recirculated) flow  M_R (t/h)
  4. Product fineness - Blaine proxy (cm^2/g)

USAGE
-----
    python3 realtime_dashboard.py

This opens an interactive matplotlib window. Drag the sliders at the
bottom while the simulation runs; the charts update live.

NOTE ON BACKENDS: this needs an interactive matplotlib backend (a GUI,
not the headless "Agg" backend used for saving static PNGs). If the
default backend on your machine doesn't pop up a window, install one of:
    pip install PyQt5        # then the script will auto-select Qt5Agg
    (Tk is usually already bundled with Python on Windows/Mac)
and re-run.
"""
import sys
import os
from collections import deque

import numpy as np
import matplotlib

# Pick an interactive backend. We check for the underlying GUI toolkit's
# presence (cheap import check) *before* importing matplotlib.pyplot, since
# switching backends after pyplot has already bound one is unreliable.
import importlib.util as _ilu

_candidates = [
    ("TkAgg", "tkinter"),
    ("Qt5Agg", "PyQt5"),
    ("QtAgg", "PyQt6"),
]
if sys.platform == "darwin":
    _candidates.append(("MacOSX", None))
_chosen_backend = None
for _backend, _module_name in _candidates:
    if _module_name is not None and _ilu.find_spec(_module_name) is None:
        continue
    try:
        matplotlib.use(_backend)
        _chosen_backend = _backend
        break
    except Exception:
        continue

if _chosen_backend is None:
    print("WARNING: no interactive GUI backend (Tk/Qt) was found.\n"
          "         Install one, e.g.:  pip install PyQt5\n"
          "         (Tk is normally bundled with Python already on Windows/Mac.)\n"
          "         The live window will not appear until one is installed.",
          file=sys.stderr)

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from cement_mill_sim import CementMillSimulator


# ----------------------------------------------------------------------
# Simulation setup
# ----------------------------------------------------------------------
sim = CementMillSimulator()

MC_DEFAULT_TH = 90.0     # t/h, initial fresh feed
DT_PER_FRAME = 1.0       # minutes of process time advanced per animation frame
FRAME_INTERVAL_MS = 150  # wall-clock ms between frames
WINDOW_MINUTES = 120.0   # rolling chart window: 2 hours of process time
HISTORY_LEN = int(WINDOW_MINUTES / DT_PER_FRAME)  # number of points kept on the strip charts

print("Warming up the simulator to an initial steady state (a few seconds)...")
_warmup = sim.simulate((0, 2000), lambda t: MC_DEFAULT_TH / 60.0,
                        t_eval=np.linspace(1900, 2000, 3))
state = _warmup.y[:, -1].copy()
t_now = 0.0   # reset the displayed clock to 0 for the interactive session

# rolling history buffers for the scrolling strip charts
t_hist = deque(maxlen=HISTORY_LEN)
H1_hist = deque(maxlen=HISTORY_LEN)
H2_hist = deque(maxlen=HISTORY_LEN)
Mr_hist = deque(maxlen=HISTORY_LEN)
blaine_hist = deque(maxlen=HISTORY_LEN)

# seed with the initial steady-state point so the charts aren't empty at t=0
_out0 = sim.instantaneous_outputs(state)
t_hist.append(t_now)
H1_hist.append(_out0["H1"])
H2_hist.append(_out0["H2"])
Mr_hist.append(_out0["Mr"])
blaine_hist.append(_out0["blaine"])


# ----------------------------------------------------------------------
# Figure / axes layout
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Cement Mill — Real-Time Simulation", fontsize=13)
plt.subplots_adjust(bottom=0.26, hspace=0.45, wspace=0.3)
ax_H1, ax_H2, ax_Mr, ax_Bl = axes.flatten()

line_H1, = ax_H1.plot([], [], color="tab:green")
ax_H1.axhline(sim.H_cap, color="firebrick", ls="--", lw=1, alpha=0.7, label="capacity limit")
ax_H1.set_title("Compartment 1 Hold-up")
ax_H1.set_xlabel("time (min)")
ax_H1.set_ylabel("H1 (t)")
ax_H1.set_ylim(0, sim.H_cap * 1.08)
ax_H1.legend(loc="upper right", fontsize=8)
ax_H1.grid(alpha=0.3)

line_H2, = ax_H2.plot([], [], color="tab:olive")
ax_H2.axhline(sim.H_cap, color="firebrick", ls="--", lw=1, alpha=0.7, label="capacity limit")
ax_H2.set_title("Compartment 2 Hold-up")
ax_H2.set_xlabel("time (min)")
ax_H2.set_ylabel("H2 (t)")
ax_H2.set_ylim(0, sim.H_cap * 1.08)
ax_H2.legend(loc="upper right", fontsize=8)
ax_H2.grid(alpha=0.3)

line_Mr, = ax_Mr.plot([], [], color="tab:purple")
ax_Mr.set_title("Return (Recirculated) Flow")
ax_Mr.set_xlabel("time (min)")
ax_Mr.set_ylabel("$M_R$ (t/h)")
ax_Mr.set_ylim(0, sim.MR_MAX * 1.05)
ax_Mr.grid(alpha=0.3)

line_Bl, = ax_Bl.plot([], [], color="tab:red")
ax_Bl.set_title("Product Fineness (Blaine proxy)")
ax_Bl.set_xlabel("time (min)")
ax_Bl.set_ylabel("cm$^2$/g")
ax_Bl.set_ylim(sim.BLAINE_MIN * 0.97, sim.BLAINE_MAX * 1.03)
ax_Bl.grid(alpha=0.3)

all_lines = [line_H1, line_H2, line_Mr, line_Bl]
all_axes = [ax_H1, ax_H2, ax_Mr, ax_Bl]
# all four charts now have guaranteed output ranges (hard cap for hold-up,
# smooth saturating maps for return flow / fineness - see cement_mill_sim.py),
# so every axis keeps a fixed y-range rather than autoscaling.
FIXED_YLIM_AXES = {id(ax_H1), id(ax_H2), id(ax_Mr), id(ax_Bl)}
all_hists = [H1_hist, H2_hist, Mr_hist, blaine_hist]

# ----------------------------------------------------------------------
# Sliders
# ----------------------------------------------------------------------
ax_feed = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_sep = plt.axes([0.15, 0.06, 0.7, 0.03])

slider_feed = Slider(ax_feed, "Fresh Feed $M_C$ (t/h)", 40.0, 150.0,
                      valinit=MC_DEFAULT_TH, color="tab:blue")
slider_sep = Slider(ax_sep, "Separator Speed (rpm)", 0.0, 750.0,
                     valinit=sim.nominal_rpm, color="tab:purple")


# ----------------------------------------------------------------------
# Real-time update loop
# ----------------------------------------------------------------------
def update(_frame):
    global state, t_now

    Mc_th = slider_feed.val
    sim.set_rotor_speed(slider_sep.val)    # live separator rotor-speed control
                                            # (higher rpm -> finer cut point, more recirculation)
    Mc0 = Mc_th / 60.0                     # t/min

    # advance the simulation by one frame's worth of process time under the
    # current (slider-set) piecewise-constant controls
    state = sim.step(state, dt=DT_PER_FRAME, Mc=Mc0)
    t_now += DT_PER_FRAME

    out = sim.instantaneous_outputs(state)
    t_hist.append(t_now)
    H1_hist.append(out["H1"])
    H2_hist.append(out["H2"])
    Mr_hist.append(out["Mr"])
    blaine_hist.append(out["blaine"])

    for line, hist in zip(all_lines, all_hists):
        line.set_data(t_hist, hist)

    for ax, hist in zip(all_axes, all_hists):
        ax.set_xlim(t_hist[0], max(t_hist[-1], t_hist[0] + 1))
        if id(ax) in FIXED_YLIM_AXES:
            continue  # hold-up charts keep their fixed 0-to-capacity y-range
        lo, hi = min(hist), max(hist)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)

    return all_lines


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Import here so the script still works (minus live animation) in
    # environments where FuncAnimation import needs the backend set first.
    from matplotlib.animation import FuncAnimation

    ani = FuncAnimation(fig, update, interval=FRAME_INTERVAL_MS, blit=False,
                         cache_frame_data=False)
    plt.show()
