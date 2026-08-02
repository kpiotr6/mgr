"""
Run the cement grinding circuit simulator and plot results:

  Figure 1 - Dynamic step response (mirrors paper Fig. 6):
             an 8% step change in the fresh feed rate M_C, showing the
             resulting transients in M_C, mill flow rate M_M, mill hold-up H,
             and product fineness (specific-surface proxy).

  Figure 2 - Steady-state operating map (mirrors paper Figs. 7-8):
             flow rates (M_M, M_P, M_R) and product rate vs. mill hold-up H,
             obtained by running the simulator to steady state at a sweep of
             fresh-feed rates. This reveals the stable operating region and
             the flooding instability at high throughput described in the
             paper (Section IV).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure cement_mill_sim.py (expected in the same folder as this script)
# is importable no matter what directory this script is run from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from cement_mill_sim import CementMillSimulator

# Save output figures next to this script too.
OUTPUT_DIR = SCRIPT_DIR


def run_step_response():
    sim = CementMillSimulator()

    Mc_base = 90.0 / 60.0     # t/min, baseline fresh feed ~90 t/h
    step_time = 300.0         # min
    step_frac = -0.08         # -8% step, as in paper Fig. 6

    def Mc_func(t):
        return Mc_base * (1.0 + step_frac) if t >= step_time else Mc_base

    t_end = 700.0
    t_eval = np.linspace(0, t_end, 400)

    # warm up to steady state at the baseline feed rate first, then apply the step
    warmup = sim.simulate((0, step_time), lambda t: Mc_base,
                           t_eval=np.linspace(0, step_time, 50))
    y0 = warmup.y[:, -1]

    sol = sim.simulate((step_time, t_end), Mc_func, y0=y0,
                        t_eval=np.linspace(step_time, t_end, 300))
    out = sim.derived_outputs(sol)

    # stitch the warm-up and post-step segments together for a full timeline plot
    warm_out = sim.derived_outputs(warmup)
    t_full = np.concatenate([warm_out['t'], out['t']])
    H_full = np.concatenate([warm_out['H'], out['H']])
    Mm_full = np.concatenate([warm_out['Mm'], out['Mm']])
    Mp_full = np.concatenate([warm_out['Mp'], out['Mp']])
    Mr_full = np.concatenate([warm_out['Mr'], out['Mr']])
    blaine_full = np.concatenate([warm_out['blaine'], out['blaine']])
    Mc_full = np.array([Mc_func(t) * 60.0 for t in t_full])  # t/h

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("Cement Mill Step Response — 8% step decrease in fresh feed rate $M_C$\n"
                 "(cf. Boulvin et al. 2003, Fig. 6)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(t_full, Mc_full, color="tab:blue")
    ax.axvline(step_time, color="gray", ls="--", lw=1)
    ax.set_ylabel("$M_C$ (t/h)")
    ax.set_title("Fresh feed rate")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_full, Mm_full, color="tab:orange")
    ax.axvline(step_time, color="gray", ls="--", lw=1)
    ax.set_ylabel("$M_M$ (t/h)")
    ax.set_title("Mill flow rate")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_full, H_full, color="tab:green")
    ax.axvline(step_time, color="gray", ls="--", lw=1)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("H (t)")
    ax.set_title("Mill hold-up")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_full, blaine_full, color="tab:red")
    ax.axvline(step_time, color="gray", ls="--", lw=1)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("specific surface (cm$^2$/g, proxy)")
    ax.set_title("Product fineness")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_step_response.png"), dpi=150)
    plt.close(fig)


def run_steady_state_sweep():
    sim = CementMillSimulator()
    Mc_values_th = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145])
    H_list, Mm_list, Mp_list, Mr_list = [], [], [], []

    y0 = None
    for Mc_th in Mc_values_th:
        Mc0 = Mc_th / 60.0

        def Mc_func(t, Mc0=Mc0):
            return Mc0

        t_final = 2600.0
        sol = sim.simulate((0, t_final), Mc_func, y0=y0,
                            t_eval=np.linspace(t_final - 100, t_final, 5))
        out = sim.derived_outputs(sol)
        H_list.append(out['H'][-1])
        Mm_list.append(out['Mm'][-1])
        Mp_list.append(out['Mp'][-1])
        Mr_list.append(out['Mr'][-1])
        # warm-start the next (higher) feed rate from this steady state, when stable
        if out['H'][-1] < 100:
            y0 = sol.y[:, -1]
        else:
            y0 = None

    H_arr, Mm_arr, Mp_arr, Mr_arr = map(np.array, (H_list, Mm_list, Mp_list, Mr_list))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("Steady-State Operating Map (cf. Boulvin et al. 2003, Figs. 7-8)", fontsize=12)

    ax = axes[0]
    ax.plot(H_arr, Mm_arr, "o-", label="$M_M$ (mill flow)")
    ax.plot(H_arr, Mp_arr, "s-", label="$M_P$ (product)")
    ax.plot(H_arr, Mr_arr, "^-", label="$M_R$ (recirculated)")
    ax.set_xlabel("mill hold-up H (t)")
    ax.set_ylabel("flow rate (t/h)")
    ax.set_title("Flow rates vs. hold-up")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(H_arr, Mp_arr, "s-", color="tab:red")
    ax.set_xlabel("mill hold-up H (t)")
    ax.set_ylabel("$M_P$ (t/h)")
    ax.set_title("Product rate vs. hold-up\n(flooding instability at high H)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_steady_state_map.png"), dpi=150)
    plt.close(fig)

    return H_arr, Mm_arr, Mp_arr, Mr_arr


if __name__ == "__main__":
    print("Running step-response simulation (Figure 1)...")
    run_step_response()
    print("Running steady-state sweep (Figure 2)...")
    H_arr, Mm_arr, Mp_arr, Mr_arr = run_steady_state_sweep()
    for h, mm, mp, mr in zip(H_arr, Mm_arr, Mp_arr, Mr_arr):
        print(f"H={h:7.2f} t   Mm={mm:8.1f} t/h   Mp={mp:6.1f} t/h   Mr={mr:8.1f} t/h")
    print("Done. Figures saved to fig1_step_response.png and fig2_steady_state_map.png")
