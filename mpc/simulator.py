import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
# https://gemini.google.com/share/1dc2d1b2157f
# --- System Parameters ---
k1 = 0.5
k2 = 0.3
dt = 0.1
history_length = 200

# --- Initial States ---
H1 = 200.0
H2 = 333.0
time = 0.0

# Pre-fill data arrays
t_data = np.linspace(-history_length*dt, 0, history_length).tolist()
H1_data = [H1] * history_length
H2_data = [H2] * history_length
R_data = [0] * history_length
B_data = [3100] * history_length

# --- GUI Figure Setup ---
# Changed figsize to (10, 8) to fit standard monitors better
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Increased bottom margin to 25% to make room for sliders without crushing charts
plt.subplots_adjust(bottom=0.25, hspace=0.6, top=0.95)
fig.canvas.manager.set_window_title('Cement Mill Simulator with Blaine')

# Initialize line objects
line_h1, = ax1.plot(t_data, H1_data, label='Comp 1 Holdup ($H_1$)', color='blue')
line_h2, = ax1.plot(t_data, H2_data, label='Comp 2 Holdup ($H_2$)', color='orange')
line_r, = ax2.plot(t_data, R_data, label='Separator Return ($R$)', color='green')
line_b, = ax3.plot(t_data, B_data, label='Blaine Fineness', color='purple')

# Configure Axis 1 (Holdups)
ax1.set_title('Compartment Holdups')
ax1.set_ylabel('Mass (tons)')
ax1.legend(loc='upper left', fontsize='small')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(0, 800)

# Configure Axis 2 (Return)
ax2.set_title('Separator Return')
ax2.set_ylabel('Flow (t/h)')
ax2.legend(loc='upper left', fontsize='small')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(0, 250)

# Configure Axis 3 (Blaine)
ax3.set_title('Product Blaine Fineness')
ax3.set_xlabel('Time')
ax3.set_ylabel('cm²/g')
ax3.legend(loc='upper left', fontsize='small')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.set_ylim(2500, 4500)

# --- Interactive Sliders ---
# Positioned safely in the bottom 25% margin we created
ax_feed = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_sep = plt.axes([0.15, 0.05, 0.7, 0.03])

slider_feed = Slider(ax_feed, 'Fresh Feed ($F$)', 0.0, 150.0, valinit=50.0)
slider_sep = Slider(ax_sep, 'Separator Speed', 0.0, 100.0, valinit=50.0)

# --- Real-Time Update Function ---
def update(frame):
    global H1, H2, time, t_data, H1_data, H2_data, R_data, B_data

    # 1. Read current control inputs
    F = slider_feed.val
    S = slider_sep.val

    # 2. Calculate Separator Dynamics
    alpha = 0.2 + (S / 100.0) * 0.6

    # 3. Calculate internal flows
    P1 = k1 * H1
    P2 = k2 * H2
    R_clean = alpha * P2

    # 4. Integrate Differential Equations (Euler Method)
    dH1 = F + R_clean - P1
    dH2 = P1 - P2

    H1 += dH1 * dt
    H2 += dH2 * dt
    time += dt

    # 5. Calculate Blaine
    B_clean = 2200 + (S * 15) + ((H1 + H2) / (F + 1.0) * 25)

    # 6. Add Gaussian Noise
    noise_std = 1.2
    H1_noisy = max(0, H1 + np.random.normal(0, noise_std))
    H2_noisy = max(0, H2 + np.random.normal(0, noise_std))
    R_noisy = max(0, R_clean + np.random.normal(0, noise_std * 0.5))
    B_noisy = B_clean + np.random.normal(0, 15.0)

    # 7. Update data histories
    t_data.append(time)
    t_data.pop(0)

    H1_data.append(H1_noisy)
    H1_data.pop(0)

    H2_data.append(H2_noisy)
    H2_data.pop(0)

    R_data.append(R_noisy)
    R_data.pop(0)

    B_data.append(B_noisy)
    B_data.pop(0)

    # 8. Update plot lines
    line_h1.set_data(t_data, H1_data)
    line_h2.set_data(t_data, H2_data)
    line_r.set_data(t_data, R_data)
    line_b.set_data(t_data, B_data)

    # 9. Shift X-axis to create scrolling effect
    ax1.set_xlim(t_data[0], t_data[-1])
    ax2.set_xlim(t_data[0], t_data[-1])
    ax3.set_xlim(t_data[0], t_data[-1])

    return line_h1, line_h2, line_r, line_b

# --- Run the Simulator ---
ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

plt.show()