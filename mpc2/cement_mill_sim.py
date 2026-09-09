"""
Cement Grinding Circuit Simulator
==================================
Based on the nonlinear distributed-parameter model described in:
Boulvin, Vande Wouwer, Lepore, Renotte, Remy,
"Modeling and Control of Cement Grinding Processes",
IEEE Trans. Control Systems Technology, 11(5), 2003.

The circuit: fresh feed (M_C) + recirculated reject (M_R) = total mill feed (M_F)
    -> ball mill (population-balance / breakage model, discretized along the
       mill axis by finite volumes -> "method of lines")
    -> mill product (M_M) -> air separator -> fines (M_P, leaves circuit)
                                            -> coarse reject (M_R, recirculated)

Key nonlinearities reproduced from the paper:
  1. Breakage rate s_i depends on the *local material hold-up* H:
         s_i(z_i, H) = a * z_i^alpha * exp(-k_s * H)          (paper eq. 16)
  2. The air-separator cut-point z50c shifts with the mill flow rate M_M,
     which is the mechanism responsible for the closed-loop instability
     and the bell-shaped M_P(H) steady-state curve (paper Figs. 5, 7, 8).

NOTE ON FIDELITY: the paper does not publish every numerical constant
needed for an exact byte-for-byte reproduction (e.g. axial diffusion
coefficient, transport velocities per compartment, full separator A-D
constants, grate classification factors). Representative values in the
same order of magnitude as those quoted in the paper (Table I/II, and
the "crash test" velocities u1=1.35, u2=0.82 m/min) are used here so
that the *qualitative* dynamic behavior described in the paper -
nonlinear bell-shaped M_P vs H, mill emptying under a negative feed
step, separator cut-point sensitivity to flow - is faithfully reproduced.
"""

import numpy as np
from scipy.integrate import solve_ivp


class CementMillSimulator:
    def __init__(self,
                 n_classes=10,       # number of particle-size classes N
                 n_cells=12,         # axial finite-volume cells along the mill
                 length=10.0,        # mill length (m)
                 u=1.1,              # transport (convection) velocity (m/min)
                 diff=0.35,          # axial diffusion coefficient (m^2/min)
                 z_min=1e-3,         # finest size (mm)
                 z_max=10.0,         # coarsest size (mm)
                 a=2.0, alpha=0.95, beta=0.9,    # breakage-law params (Table II order of magnitude)
                 k_s=0.2,            # hold-up sensitivity of breakage rate (eq. 16)
                 z50c_base=0.09,     # base separator cut point (mm) at reference mill flow
                 Mm_ref=90.0,        # reference mill flow rate for cut-point correlation (t/h)
                 p_cutpoint=0.6,     # sensitivity of cut point to mill flow rate
                 compartment_split=None,   # axial cell index dividing compartment 1 / 2
                                            # (defaults to the midpoint, as in the paper's
                                            # two-compartment mill, Fig. 1)
                 compartment_capacity=100.0,   # physical hold-up limit per compartment (t)
                 overflow_gain=6.0,      # strength of the overflow/relief response
                                          # (higher = harder cap, stiffer near the limit)
                 sep_sharpness=3.0,      # partition-curve exponent m in E = f_min +
                                          # (f_max-f_min)/(1+z_r^m); higher = sharper cut
                 nominal_rpm=150.0,      # rotor speed at which sep_bias=1 (z50c = z50c_base)
                 speed_cutpoint_exponent=1.5,  # cut point ~ (nominal_rpm/rpm)^exponent
                                                # (higher speed -> smaller cut point -> finer product)
                 speed_sharpness_exponent=0.3):  # higher speed also sharpens the separation curve
        """
        nominal_rpm / speed_cutpoint_exponent / speed_sharpness_exponent configure
        the *rotor-speed* control path (see set_rotor_speed()), used for modeling
        modern dynamic/high-efficiency separators (SEPOL, O-Sepa, Sepmaster-style),
        as opposed to the register-vane control path (sep_bias, set directly) used
        for older static separators like the one in the Boulvin et al. paper.
        Both paths ultimately act through the same sep_bias multiplier on z50c,
        so you can use either interchangeably, but not meaningfully at once.
        """
        self.N = n_classes
        self.Nx = n_cells
        self.L = length
        self.dx = length / n_cells
        self.D = diff
        self.k_s = k_s
        self.z50c_base = z50c_base
        self.Mm_ref = Mm_ref
        self.p_cutpoint = p_cutpoint
        # separator register bias: >1 = registers more closed -> coarser cut
        # point (more material rejected); <1 = more open -> finer cut point.
        # This is the live control exposed to the "Separator Register" slider.
        self.sep_bias = 1.0
        self.H_cap = compartment_capacity
        self.overflow_gain = overflow_gain

        # --- rotor-speed control path (dynamic/high-efficiency separator) ---
        self.nominal_rpm = nominal_rpm
        self.speed_cutpoint_exponent = speed_cutpoint_exponent
        self.speed_sharpness_exponent = speed_sharpness_exponent
        self.sep_sharpness = sep_sharpness   # partition exponent m, see separator_efficiency()
        self._sep_sharpness0 = sep_sharpness  # base value, for the disabled speed coupling

        # Geometric size grid, class 0 = coarsest, class N-1 = finest
        R = (z_max / z_min) ** (1.0 / (self.N - 1))
        self.z = z_max / (R ** np.arange(self.N))          # upper limit of each class
        self.a, self.alpha, self.beta = a, alpha, beta

        # Cumulative breakage distribution B_ij = (z_{i-1}/z_j)^beta,  j < i
        # (probability that debris from class j lands at class <= i)
        self.b = self._breakage_matrix()

        split = compartment_split if compartment_split is not None else n_cells // 2
        self.compartment_split = split
        a_cell = np.empty(self.Nx)
        u_cell_base = np.empty(self.Nx)
        a_cell[:split] = a * 1.3     # compartment 1: faster breakage
        a_cell[split:] = a * 0.7     # compartment 2: slower breakage
        u_cell_base[:split] = u * 1.25    # compartment 1: faster transport
        u_cell_base[split:] = u * 0.75    # compartment 2: slower transport
        self.a_cell = a_cell
        self.u_cell_base = u_cell_base   # nominal velocities; see _effective_u_cell()

        # Fresh-feed particle-size distribution: clinker is coarse -> weight toward
        # the largest classes (simple triangular-ish profile)
        w = np.linspace(1.0, 0.05, self.N)
        self.mC = w / w.sum()

        # state layout: W[k, i] = mass of size class i held in axial cell k  (tons)
        self.state_shape = (self.Nx, self.N)

        # calibration constant for the fineness proxy, set so typical operating
        # points land in a realistic Blaine range (see derived_outputs())
        self._blaine_cal = 22.0

        # --- output-range shaping ---
        # The raw fineness/recirculation numbers computed from the model can
        # swing very widely at extreme slider settings (e.g. deep in the
        # overflow/flooding regime). Rather than hard-clipping - which would
        # just discard information - both readouts are passed through a
        # smooth saturating map: near-identity in the normal operating
        # region, asymptotically approaching (but never exceeding) the given
        # bounds at extremes. Tune these constants to change the target
        # ranges shown on the dashboard/plots.
        self.BLAINE_MIN, self.BLAINE_MAX = 2200.0, 7200.0
        self.BLAINE_CENTER, self.BLAINE_SCALE = 4500.0, 900.0   # logistic map shape
        self.MR_MAX, self.MR_TAU = 300.0, 300.0                  # exponential-saturation map shape

    # ------------------------------------------------------------------
    def _map_blaine(self, raw):
        """Smoothly map the raw fineness metric into [BLAINE_MIN, BLAINE_MAX]
        via a logistic curve: near-identity around BLAINE_CENTER, saturating
        toward the bounds far from it."""
        x = (raw - self.BLAINE_CENTER) / self.BLAINE_SCALE
        return self.BLAINE_MIN + (self.BLAINE_MAX - self.BLAINE_MIN) / (1.0 + np.exp(-x))

    def _map_return_flow(self, raw):
        """Smoothly map the raw recirculated-flow rate into [0, MR_MAX] via
        exponential saturation: near-identity for typical values, asymptotic
        toward MR_MAX for large (e.g. flooding-regime) values."""
        return self.MR_MAX * (1.0 - np.exp(-max(raw, 0.0) / self.MR_TAU))

    # ------------------------------------------------------------------
    def _breakage_matrix(self):
        """b[i,j] = fraction of material broken out of class j that reappears
        in (finer) class i, for i>j. z is sorted coarsest (index 0) -> finest
        (index N-1). Uses the power-law cumulative breakage distribution
        B(z) = (z / z_j)^beta of eq. (5)/(15) in the paper: the fraction of
        debris from a parent of size z_j that ends up finer than a given
        boundary z. The finest class acts as an absorbing catch-all bin for
        anything ground below the smallest size boundary."""
        N = self.N
        z = self.z
        b = np.zeros((N, N))
        for j in range(N - 1):          # the finest class (N-1) cannot break further
            prev_cdf = 1.0               # CDF just below the parent's own boundary
            for i in range(j + 1, N):
                if i < N - 1:
                    cdf_i = (z[i] / z[j]) ** self.beta   # fraction finer than boundary z[i]
                else:
                    cdf_i = 0.0                           # finest bin absorbs everything below it
                b[i, j] = max(prev_cdf - cdf_i, 0.0)
                prev_cdf = cdf_i
        colsum = b.sum(axis=0)
        for j in range(N):
            if colsum[j] > 0:
                b[:, j] /= colsum[j]
        # column mask: True where breaking material out of class j has somewhere
        # to go (i.e. j is not the finest class). Prevents a mass leak where the
        # finest class would otherwise "break" into nothing.
        self.breakable = colsum > 1e-12
        return b

    # ------------------------------------------------------------------
    def set_rotor_speed(self, rpm):
        """Live control for a *dynamic* (rotor-speed-controlled) separator.
        Increasing rotor speed increases the centrifugal force on particles
        in the classifying zone, throwing more material - including
        moderately fine particles - back to the reject stream. Net effect:
        higher speed -> smaller (finer) cut point, more recirculation,
        finer product; lower speed -> coarser cut point, less recirculation,
        coarser product. This is the opposite control sense from opening/
        closing register vanes on a static separator.

        Derived from a simple force-balance approximation (centrifugal force
        ~ rotor_speed^2 * particle_mass, balanced against aerodynamic drag),
        which gives cut size d_cut ~ 1/rotor_speed to first order; the
        `speed_cutpoint_exponent` lets you tune that sensitivity.
        Higher speed also sharpens the separation curve somewhat (better
        classification efficiency at speed), reflected in `speed_sharpness_exponent`.
        """
        rpm = max(rpm, 1.0)
        ratio = self.nominal_rpm / rpm
        self.sep_bias = ratio ** self.speed_cutpoint_exponent
        # Disabled: the speed->sharpness coupling is not supported by the
        # separator literature, which ties sharpness to classifier *load*
        # (higher throughput -> less sharp; Altun & Benzer 2014, Fig. 16)
        # rather than to rotor speed. Re-enable only against a base value,
        # never the old hardcoded 3.0, or it would clobber sep_sharpness:
        # self.sep_sharpness = self._sep_sharpness0 * (rpm / self.nominal_rpm) ** self.speed_sharpness_exponent

    # ------------------------------------------------------------------
    def separator_cutpoint(self, Mm):
        """Cut point grows as mill flow rate falls (paper Sec. IV / Fig. 5),
        and is scaled by the live `sep_bias` register-position control (>1 =
        registers more closed -> coarser cut, more recirculation; <1 = more
        open -> finer cut, less recirculation)."""
        Mm = max(Mm, 1.0)
        return self.z50c_base * self.sep_bias * (self.Mm_ref / Mm) ** self.p_cutpoint

    def separator_efficiency(self, Mm):
        """Fraction of each size class reporting to the *fine product* stream.
        Implemented as a bounded, monotonically-decreasing partition (Tromp-type)
        curve around the cut point z50c: fine particles mostly exit as product,
        coarse particles are mostly rejected and recirculated. This is a
        numerically robust stand-in for the paper's eq. (10) reduced-efficiency
        curve (same qualitative S-shape, guaranteed to stay in [f_min, f_max]).

        Uses the *log-logistic* partition form

            E(z_r) = f_min + (f_max - f_min) / (1 + z_r^m)

        i.e. a logistic in log(z_r) rather than in z_r itself. This is the
        standard shape of a published partition curve (cf. the Plitt /
        Lynch-Rao / logistic family used for classifiers, and Altun & Benzer,
        Powder Technology 264 (2014) 1-8, whose measured d50c range of
        0.03-0.11 mm brackets z50c_base), and it fixes two defects of the
        plain logistic in z_r used previously:

          * f_max is now *attained* as z_r -> 0, so the fine-end plateau -
            and hence the separator bypass, 1 - f_max - is exactly what the
            constant says, instead of drifting with the sharpness;
          * the curve is symmetric on the log-size axis, which is how
            separation curves are measured and plotted (paper Figs. 2, 4, 5).

        Evaluated as exp(m*ln z_r) so the same defensive clipping as before
        keeps very fine / very coarse classes from over- or underflowing.
        """
        z50c = self.separator_cutpoint(Mm)
        zr = self.z / z50c
        m = self.sep_sharpness        # partition exponent; higher = sharper cut
        f_min, f_max = 0.03, 0.92     # residual coarse escape / fine bypass floors
        exponent = np.clip(m * np.log(np.maximum(zr, 1e-300)), -50, 50)
        E = f_min + (f_max - f_min) / (1.0 + np.exp(exponent))
        return E

    # ------------------------------------------------------------------
    def rhs(self, t, y, Mc_func):
        """Right-hand side of the full ODE system (mill axial cells + recirculation)."""
        W = y.reshape(self.state_shape)          # (Nx, N) tons per cell
        H_cell = W.sum(axis=1)                    # hold-up per cell (t)

        H1, H2 = self.compartment_holdups(W)
        u_cell = self._effective_u_cell(H1, H2)   # overflow-relief-adjusted transport velocity

        # --- mill product leaving the last cell (advective outflow, t/min per class) ---
        # NOTE: this must use the same flux expression as the last cell's
        # outflow term below ((u_cell/dx)*W) or the mass balance will not close.
        Mm_i = (u_cell[-1] / self.dx) * W[-1, :]
        Mm_total = Mm_i.sum()  # t/min

        # --- separator split ---
        E = self.separator_efficiency(Mm_total * 60.0)   # efficiency curve keyed off t/h
        Mp_i = Mm_i * E
        Mr_i = Mm_i * (1 - E)
        Mr_total = Mr_i.sum()

        # --- fresh feed & combined mill feed ---
        Mc_total = Mc_func(t)  # t/min
        Mc_i = Mc_total * self.mC
        Mf_i = Mc_i + Mr_i

        # vectorized breakage rates: shape (Nx, N). Note: the eq.(16) nonlinearity
        # s_i ~ exp(-k_s*H) uses H as a *linear hold-up density* (t/m, paper's Table
        # values are ~1-3 t/m), not the absolute mass held in a finite-volume cell.
        # Each cell uses its own compartment's breakage-rate scale (a_cell).
        H_density = H_cell / self.dx
        s = np.zeros((self.Nx, self.N))
        for k in range(self.Nx):
            s[k, :] = self.a_cell[k] * (self.z ** self.alpha) * np.exp(-self.k_s * max(H_density[k], 0.0))
        s = s * self.breakable[np.newaxis, :]   # finest class cannot break further (mass conservation)

        breakage_sink = s * W
        # breakage source: sum_j b[i,j] * s_j(H) * W_j   (per cell)
        breakage_source = np.einsum('ij,kj->ki', self.b, s * W)

        # --- convection (upwind, per-cell velocity) + diffusion (central) along the mill axis ---
        conv = np.zeros_like(W)
        diff = np.zeros_like(W)
        for k in range(self.Nx):
            flux_in = Mf_i if k == 0 else (u_cell[k - 1] / self.dx) * W[k - 1, :]
            flux_out = (u_cell[k] / self.dx) * W[k, :]
            conv[k, :] = flux_in - flux_out

            left = W[k - 1, :] if k > 0 else W[k, :]
            right = W[k + 1, :] if k < self.Nx - 1 else W[k, :]
            diff[k, :] = self.D / self.dx ** 2 * (left - 2 * W[k, :] + right)

        dWdt = conv + diff - breakage_sink + breakage_source
        return dWdt.flatten()

    # ------------------------------------------------------------------
    def compartment_holdups(self, W_or_y):
        """Split the axial cells into two compartments (as in the paper's
        two-compartment mill, Fig. 1) and return (H1, H2) in tons.
        Accepts either the (Nx, N) state array or the flattened state vector."""
        W = W_or_y.reshape(self.state_shape) if W_or_y.ndim == 1 else W_or_y
        H_cell = W.sum(axis=1)
        split = self.compartment_split
        H1 = H_cell[:split].sum()
        H2 = H_cell[split:].sum()
        return H1, H2

    # ------------------------------------------------------------------
    def step(self, y0, dt, Mc):
        """Advance the state by dt (minutes) under a constant fresh-feed rate
        Mc (t/min), using a lighter-weight integrator tuned for repeated,
        short calls (e.g. one animation frame in a real-time dashboard).
        Returns the new flattened state vector."""
        sol = solve_ivp(self.rhs, (0.0, dt), y0, args=(lambda t: Mc,),
                         method='RK45', rtol=1e-5, atol=1e-7, max_step=dt)
        return sol.y[:, -1]

    # ------------------------------------------------------------------
    def _effective_u_cell(self, H1, H2):
        """Per-cell transport velocity, boosted by an overflow/relief response
        once a compartment's hold-up approaches its physical capacity
        (self.H_cap tonnes). This mimics a mill discharging faster as its
        material level rises toward the grate/overflow limit, which keeps
        hold-up from growing without bound - similar in spirit to a real
        mill's overflow trunnion or discharge diaphragm. The response ramps
        up smoothly starting at 85% of capacity and grows steeply (cubic)
        beyond it, so the compartment settles at or just above the cap
        instead of flooding indefinitely."""
        cap = self.H_cap
        threshold = 0.85 * cap

        def relief_multiplier(H):
            excess = max(0.0, (H - threshold) / (0.15 * cap))
            return 1.0 + self.overflow_gain * excess ** 3

        split = self.compartment_split
        u_eff = np.empty(self.Nx)
        u_eff[:split] = self.u_cell_base[:split] * relief_multiplier(H1)
        u_eff[split:] = self.u_cell_base[split:] * relief_multiplier(H2)
        return u_eff

    # ------------------------------------------------------------------
    def simulate(self, t_span, Mc_func, y0=None, t_eval=None):
        if y0 is None:
            # start from a modest, evenly distributed hold-up
            W0 = np.zeros(self.state_shape)
            W0[:, :] = 0.15
            y0 = W0.flatten()
        sol = solve_ivp(self.rhs, t_span, y0, args=(Mc_func,),
                         method='BDF', t_eval=t_eval, rtol=1e-6, atol=1e-8,
                         max_step=1.0)
        return sol

    # ------------------------------------------------------------------
    def instantaneous_outputs(self, y):
        """Compute the scalar quantities of interest (Mm, Mp, Mr, blaine, H1, H2)
        directly from a raw (flattened) state vector - handy for a real-time
        loop where you only have the current state, not a full solve_ivp
        solution object."""
        W = y.reshape(self.state_shape)
        H1, H2 = self.compartment_holdups(W)
        u_cell = self._effective_u_cell(H1, H2)
        Mm_i = (u_cell[-1] / self.dx) * W[-1, :]
        Mm = Mm_i.sum() * 60.0  # t/h
        E = self.separator_efficiency(Mm)
        Mp_i = Mm_i * E
        Mr_i = Mm_i * (1 - E)
        Mp = Mp_i.sum() * 60.0
        Mr_raw = Mr_i.sum() * 60.0
        blaine_raw = 0.0
        if Mp_i.sum() > 1e-9:
            mp_frac = Mp_i / Mp_i.sum()
            blaine_raw = self._blaine_cal * np.sum(mp_frac / self.z)
        Mr = self._map_return_flow(Mr_raw)
        blaine = self._map_blaine(blaine_raw)
        return dict(Mm=Mm, Mp=Mp, Mr=Mr, blaine=blaine, H1=H1, H2=H2, H=H1 + H2,
                    Mr_raw=Mr_raw, blaine_raw=blaine_raw)

    # ------------------------------------------------------------------
    def derived_outputs(self, sol):
        """Post-process a solve_ivp solution into the physical quantities
        used for plotting: H(t), M_M(t), M_P(t), M_R(t), specific surface(t)."""
        n_t = sol.t.size
        H = np.zeros(n_t)
        Mm = np.zeros(n_t)
        Mp = np.zeros(n_t)
        Mr = np.zeros(n_t)
        blaine = np.zeros(n_t)
        for idx in range(n_t):
            W = sol.y[:, idx].reshape(self.state_shape)
            H[idx] = W.sum()
            H1_, H2_ = self.compartment_holdups(W)
            u_cell = self._effective_u_cell(H1_, H2_)
            Mm_i = (u_cell[-1] / self.dx) * W[-1, :]
            Mm_total = Mm_i.sum()
            Mm[idx] = Mm_total * 60.0  # t/h
            E = self.separator_efficiency(Mm[idx])
            Mp_i = Mm_i * E
            Mr_i = Mm_i * (1 - E)
            Mp[idx] = Mp_i.sum() * 60.0
            Mr[idx] = Mr_i.sum() * 60.0
            # Fineness proxy (relative specific-surface indicator, cm^2/g scale):
            # proportional to the mass-weighted sum of 1/size across the product
            # stream. This is a simplified stand-in for the paper's measured
            # specific surface, not a first-principles surface-area calculation.
            if Mp_i.sum() > 1e-9:
                mp_frac = Mp_i / Mp_i.sum()
                blaine[idx] = self._blaine_cal * np.sum(mp_frac / self.z)
        Mr_raw, blaine_raw = Mr.copy(), blaine.copy()
        Mr = self.MR_MAX * (1.0 - np.exp(-np.clip(Mr_raw, 0, None) / self.MR_TAU))
        x = (blaine_raw - self.BLAINE_CENTER) / self.BLAINE_SCALE
        blaine = self.BLAINE_MIN + (self.BLAINE_MAX - self.BLAINE_MIN) / (1.0 + np.exp(-x))
        return dict(t=sol.t, H=H, Mm=Mm, Mp=Mp, Mr=Mr, blaine=blaine,
                    Mr_raw=Mr_raw, blaine_raw=blaine_raw)
