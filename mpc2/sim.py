import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

# --- 1. System Parameters ---
N = 15          # Number of size intervals
M = 11          # Spatial discretization nodes (3 in comp 1, 8 in comp 2)
L = 12.0        # Length of the mill (meters)
dx = L / (M - 1)
dt = 0.5        # Time step (minutes)

# Breakage parameters (Compartment 1 and 2)
alpha = [1.03, 0.85]
a = [1.16, 0.56]
k_s = 1.1

# Transport parameters (Simplified)
u = np.ones(N) * 1.35
D = np.ones(N) * 0.1

# Geometric progression sizes for N intervals
sizes = 10.0 * (0.5 ** np.arange(N))

# Separator Parameters (Representative values based on the reduced efficiency curve)
A_sep, B_sep, C_sep, D_sep = 1.0, 1.0, 1.0, 0.1

# --- 2. Core Functions ---
def calc_breakage_rate(z, H, compartment):
    comp_idx = 0 if compartment == 1 else 1
    return (a[comp_idx] * (z ** alpha[comp_idx])) * np.exp(-k_s * H)

def calc_breakage_distribution(z_i_minus_1, z_j, compartment):
    comp_idx = 0 if compartment == 1 else 1
    return (z_i_minus_1 / z_j) ** alpha[comp_idx]

# Precompute breakage distributions
B_ij_comp1 = np.zeros((N, N))
B_ij_comp2 = np.zeros((N, N))
for i in range(1, N):
    for j in range(i):
        B_ij_comp1[i, j] = calc_breakage_distribution(sizes[i-1], sizes[j], 1) - calc_breakage_distribution(sizes[i], sizes[j], 1)
        B_ij_comp2[i, j] = calc_breakage_distribution(sizes[i-1], sizes[j], 2) - calc_breakage_distribution(sizes[i], sizes[j], 2)

# --- 3. Live Plot & Slider Setup ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
plt.subplots_adjust(bottom=0.25) # Make room for sliders

# Subplot 1: Hold-ups
line_h1, = ax1.plot([], [], label='Compartment 1 Hold-up', color='blue', linewidth=2)
line_h2, = ax1.plot([], [], label='Compartment 2 Hold-up', color='orange', linewidth=2)
ax1.set_title("Mill Hold-Up Dynamics")
ax1.set_ylabel("Hold-Up (Tonnes)")
ax1.set_ylim(0, 110) # Max holdup visible to 110 tonnes to show the 100 hard stop clearly
ax1.set_xlim(0, 50)
ax1.legend(loc='upper left')
ax1.grid(True)

# Subplot 2: Return Flow (M_R)
line_mr, = ax2.plot([], [], label='Return Flow (M_R)', color='red', linewidth=2)
ax2.set_title("Separator Return Flow Dynamics")
ax2.set_xlabel("Time (minutes)")
ax2.set_ylabel("Flow Rate (t/h)")
ax2.set_ylim(0, 100)
ax2.set_xlim(0, 50)
ax2.legend(loc='upper left')
ax2.grid(True)

# Define Sliders Axes
ax_feed = plt.axes([0.15, 0.1, 0.7, 0.03])
ax_sep = plt.axes([0.15, 0.04, 0.7, 0.03])

# Create Sliders
slider_feed = Slider(ax=ax_feed, label='Fresh Feed (t/h)', valmin=0.0, valmax=100.0, valinit=0.0)
# Separator speed is represented by adjusting the cut-point z50c. Lower z50c = higher speed/finer cut.
slider_sep = Slider(ax=ax_sep, label='Sep Cut-point (mm)', valmin=0.1, valmax=1.0, valinit=0.5)

# --- 4. Real-Time Simulation Loop ---
Hm = np.zeros((M, N))
fresh_feed_fractions = np.ones(N) / N

# History arrays for plotting
time_hist = []
h1_hist, h2_hist, mr_hist = [], [], []

print("Starting interactive closed-loop simulation. Use the sliders at the bottom to adjust parameters. Close the window to stop.")

try:
    sim_time = 0.0
    M_R_fractions = np.zeros(N)
    M_R_total = 0.0

    while plt.fignum_exists(fig.number):
        # Read current values from sliders
        M_C = slider_feed.val
        z50c = slider_sep.val

        dHm_dt = np.zeros((M, N))
        H_nodes = np.sum(Hm, axis=1)

        # Calculate closed-loop feed (Fresh Feed + Recirculated Return)
        M_F = M_C + M_R_total
        if M_F > 0:
            feed_fractions = ((M_C * fresh_feed_fractions) + M_R_fractions) / M_F
        else:
            feed_fractions = fresh_feed_fractions

        # Process spatial nodes
        for x in range(M):
            comp = 1 if x < 3 else 2
            B_matrix = B_ij_comp1 if comp == 1 else B_ij_comp2

            for i in range(N):
                # Spatial transport
                if x == 0:
                    dHm_dx = (Hm[x+1, i] - Hm[x, i]) / dx
                    d2Hm_dx2 = 0
                    convection_in = M_F * feed_fractions[i] / dx
                elif x == M - 1:
                    dHm_dx = (Hm[x, i] - Hm[x-1, i]) / dx
                    d2Hm_dx2 = 0
                    convection_in = -u[i] * Hm[x, i] / dx
                else:
                    dHm_dx = (Hm[x, i] - Hm[x-1, i]) / dx
                    d2Hm_dx2 = (Hm[x+1, i] - 2*Hm[x, i] + Hm[x-1, i]) / (dx**2)
                    convection_in = 0

                transport = -u[i] * dHm_dx + D[i] * d2Hm_dx2 + convection_in

                # Breakage
                s_i = calc_breakage_rate(sizes[i], H_nodes[x], comp)
                appearance = 0
                if i > 0:
                    for j in range(i):
                        s_j = calc_breakage_rate(sizes[j], H_nodes[x], comp)
                        appearance += B_matrix[i, j] * s_j * Hm[x, j]

                breakage = -s_i * Hm[x, i] + appearance
                dHm_dt[x, i] = transport + breakage

        # Step forward in time
        Hm += dHm_dt * dt
        Hm = np.maximum(Hm, 0)

        # Calculate total hold-up per compartment (Integrate over length)
        H1_total = np.sum(np.sum(Hm[0:3, :], axis=1)) * dx
        H2_total = np.sum(np.sum(Hm[3:11, :], axis=1)) * dx

        # --- ENFORCE HARD STOP AT 100 TONNES PER COMPARTMENT ---
        if H1_total > 100.0:
            scale_factor_1 = 100.0 / H1_total
            Hm[0:3, :] *= scale_factor_1
            H1_total = 100.0

        if H2_total > 100.0:
            scale_factor_2 = 100.0 / H2_total
            Hm[3:11, :] *= scale_factor_2
            H2_total = 100.0

        # Calculate Mill Output (M_M) at the final node
        M_M_i = u * Hm[M-1, :]

        # Calculate Separator Return (M_R)
        for i in range(N):
            z_r = sizes[i] / z50c
            E_zr = (C_sep * np.exp(-D_sep * z_r)) / (A_sep * z_r**2 - B_sep * z_r + 1)
            E_zr = min(max(E_zr, 0), 1) # Clamp efficiency between 0 and 1
            M_R_fractions[i] = M_M_i[i] * (1 - E_zr) # Underflow

        M_R_total = np.sum(M_R_fractions)

        # Update histories
        time_hist.append(sim_time)
        h1_hist.append(H1_total)
        h2_hist.append(H2_total)
        mr_hist.append(M_R_total)

        # Update Plot Lines
        line_h1.set_data(time_hist, h1_hist)
        line_h2.set_data(time_hist, h2_hist)
        line_mr.set_data(time_hist, mr_hist)

        # Scroll x-axis dynamically
        if sim_time > ax1.get_xlim()[1] * 0.9:
            ax1.set_xlim(0, sim_time * 1.5)
            ax2.set_xlim(0, sim_time * 1.5)

        fig.canvas.draw()
        fig.canvas.flush_events()

        sim_time += dt

        # Enforce updates exactly every one second
        time.sleep(1.0)

except KeyboardInterrupt:
    print("Simulation stopped by user.")