import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

try:
    from mpc.simulator_headless import CementMillSim, CementMillSimConfig
except ModuleNotFoundError:  # running as a script: `python mpc/simulator.py`
    import os
    import sys

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from mpc.simulator_headless import CementMillSim, CementMillSimConfig


def main() -> None:
    cfg = CementMillSimConfig(dt=0.1)
    sim = CementMillSim(config=cfg, seed=0)

    history_length = 200
    cap = float(cfg.compartment_capacity_tonnes)

    t_data = np.linspace(-history_length * cfg.dt, 0.0, history_length).tolist()
    H1_data = [sim.H1] * history_length
    H2_data = [sim.H2] * history_length
    R_data = [0.0] * history_length
    B_data = [3100.0] * history_length

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25, hspace=0.6, top=0.95)
    fig.canvas.manager.set_window_title("Cement Mill Simulator with Blaine")

    line_h1, = ax1.plot(t_data, H1_data, label="Comp 1 Holdup ($H_1$)", color="blue")
    line_h2, = ax1.plot(t_data, H2_data, label="Comp 2 Holdup ($H_2$)", color="orange")
    line_r, = ax2.plot(t_data, R_data, label="Separator Return ($R$)", color="green")
    line_b, = ax3.plot(t_data, B_data, label="Blaine Fineness", color="purple")

    ax1.set_title("Compartment Holdups")
    ax1.set_ylabel("Mass (tons)")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim(0.0, cap * 1.2)

    ax2.set_title("Separator Return")
    ax2.set_ylabel("Flow (t/h)")
    ax2.legend(loc="upper left", fontsize="small")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_ylim(0.0, 250.0)

    ax3.set_title("Product Blaine Fineness")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("cm²/g")
    ax3.legend(loc="upper left", fontsize="small")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.set_ylim(2000.0, 7000.0)

    ax_feed = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_sep = plt.axes([0.15, 0.05, 0.7, 0.03])

    slider_feed = Slider(
        ax_feed,
        "Fresh Feed ($F$)",
        cfg.clinker_1_feedrate_min,
        cfg.clinker_1_feedrate_max,
        valinit=60.0,
    )
    slider_sep = Slider(
        ax_sep,
        "Separator Speed",
        cfg.separator_speed_min,
        cfg.separator_speed_max,
        valinit=(cfg.separator_speed_min + cfg.separator_speed_max) / 2.0,
    )

    def update(_frame):
        y = sim.step(
            clinker_1_feedrate=float(slider_feed.val),
            separator_speed=float(slider_sep.val),
        )

        H1_noisy = (float(y["first_chamber_filling"]) / 100.0) * cap
        H2_noisy = (float(y["second_chamber_filling"]) / 100.0) * cap

        t_data.append(float(y["time"]))
        t_data.pop(0)

        H1_data.append(H1_noisy)
        H1_data.pop(0)

        H2_data.append(H2_noisy)
        H2_data.pop(0)

        R_data.append(float(y["return"]))
        R_data.pop(0)

        B_data.append(float(y["gran1_blain"]))
        B_data.pop(0)

        line_h1.set_data(t_data, H1_data)
        line_h2.set_data(t_data, H2_data)
        line_r.set_data(t_data, R_data)
        line_b.set_data(t_data, B_data)

        ax1.set_xlim(t_data[0], t_data[-1])
        ax2.set_xlim(t_data[0], t_data[-1])
        ax3.set_xlim(t_data[0], t_data[-1])

        return line_h1, line_h2, line_r, line_b

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()

    # Keep a reference alive until after the GUI closes.
    _ = ani


if __name__ == "__main__":
    main()