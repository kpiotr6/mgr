# `cement_mill_sim.py` — Model Description

This document explains how `CementMillSimulator` (in `mpc2/cement_mill_sim.py`) works,
mapping each part of the code to the equations of the paper it is based on:

> M. Boulvin, A. Vande Wouwer, R. Lepore, C. Renotte, M. Remy,
> "Modeling and Control of Cement Grinding Processes,"
> *IEEE Transactions on Control Systems Technology*, 11(5), 2003, pp. 715–724.

The simulator reproduces the paper's grinding circuit **qualitatively** (correct
nonlinear mechanisms and bell-shaped steady-state curves), not as a byte-exact
numerical replica — several constants that the paper does not publish (axial
diffusion coefficient, per-compartment velocities, full separator A–D
coefficients, grate classification factors) are chosen by the code's author to be
"the same order of magnitude" as the paper's Tables I/II and its crash-test values.

---

## 1. The physical circuit

```
Mc (fresh feed) ─┐
                  ├─► Mf (total feed) ─► [ Ball mill ] ─► Mm (mill product)
Mr (reject) ──────┘                                            │
   ▲                                                            ▼
   └────────────────────────────── [ Air separator ] ◄─────────┘
                                          │
                                          ▼
                                    Mp (fine product, leaves circuit)
```

This is exactly Fig. 1 of the paper: a two-compartment ball mill in closed loop
with an air separator. `Mc` (clinker + additives) and the recirculated coarse
reject `Mr` combine into the total mill feed `Mf = Mc + Mr`. The mill product
`Mm` is split by the separator into fines `Mp` (final product) and coarse reject
`Mr` (recycled back to the mill inlet).

---

## 2. Particle population and state representation

Following Brown (1941) / Sedlatschek & Bass, the material is not treated as a
single bulk mass but as a **population of N particle-size classes**
(`n_classes`, default 10), class 0 = coarsest, class N−1 = finest:

```python
R = (z_max / z_min) ** (1.0 / (self.N - 1))
self.z = z_max / (R ** np.arange(self.N))
```

This builds a geometric size grid (constant ratio `R` between consecutive class
boundaries), matching the paper's assumption in Section II-A that "size
intervals are assumed to have a geometric progression, with a constant ratio
`R = z_j / z_{j+1}`" (text following eq. (3)).

The mill itself is treated as a **tubular reactor**, discretized along its axis
into `Nx` finite-volume cells (`n_cells`, default 12) — the paper calls this a
distributed-parameter / "method of lines" solution of the governing PDE
(Section IV: "a numerical procedure based on the method of lines … for solving
the model partial differential equations").

The simulation state is the mass of each size class in each axial cell:

```python
W[k, i]  # tons of size-class i held in axial cell k,  shape (Nx, N)
```

This corresponds to the paper's `H(x,t) m_i(x,t)` — hold-up density times mass
fraction — integrated over one finite-volume cell, i.e. `W[k,i] ≈ ∫ H(x,t) m_i(x,t) dx`
over cell `k`.

---

## 3. Mill mass balance (paper eq. (6)–(9))

The paper's governing PDE for each size class `i` is:

```
∂(H m_i)/∂t = -u_i ∂(H m_i)/∂x + D_i ∂²(H m_i)/∂x²  -  s_i H m_i  +  Σ_{j<i} b_{ij} s_j H m_j     (6)
```

i.e. convection (transport along the mill) + diffusion (axial mixing) −
breakage-out (loss from class i) + breakage-in (fragments arriving from coarser
classes j < i). The code's `rhs()` method computes exactly these four terms per
cell, discretized:

- **Convection** (`conv`): first-order upwind flux difference,
  `flux_in - flux_out = (u_{k-1}/dx) W_{k-1} - (u_k/dx) W_k`, using the
  boundary condition `flux_in = Mf_i` for the first cell (paper eq. (8): the
  feed `M_F(t) · m_{F,i}(t)` enters at `x=0`) and free outflow
  `(u_k/dx) W_k` at the last cell (paper eq. (9), with the classifying factor
  `Pr_i` from the grate simplified to zero, i.e. all product exits the mill
  and is diverted to the separator rather than being partially rejected at
  the grate itself).
- **Diffusion** (`diff`): central-difference Laplacian
  `D/dx² · (W_{k-1} - 2W_k + W_{k+1})`, with zero-flux (mirror) conditions at
  the two mill ends.
- **Breakage sink/source** (`breakage_sink`, `breakage_source`): see §4 below.

```python
dWdt = conv + diff - breakage_sink + breakage_source
```

### Two-compartment structure

The paper's mill has two compartments with different breakage/transport
parameters (its Table I/II gives, e.g., `a₁≈1.16` vs `a₂≈0.56`, and its
crash-test gives `u¹=1.35 m/min` vs `u²=0.82 m/min`). The code reproduces this
by splitting the `Nx` axial cells at `compartment_split` (default: midpoint)
and scaling the base breakage-rate constant `a` and transport velocity `u` per
compartment:

```python
a_cell[:split] = a * 1.3     # compartment 1: faster breakage (coarse grinding)
a_cell[split:] = a * 0.7     # compartment 2: slower breakage (fine grinding)
u_cell_base[:split] = u * 1.25
u_cell_base[split:] = u * 0.75
```

chosen so the *average* of `a` and `u` across the mill match the single values
passed to `__init__`, while still giving each compartment a distinct
character.

---

## 4. Breakage law: rates and distribution (paper eq. (4), (5), (16))

### Specific breakage rate `s_i`

The paper's simplified breakage law (eq. (4), obtained from the more general
eq. (1) by dropping the abnormal-breakage correction `Q(z_i)`, justified for
industrial mills without abnormal large-particle breakage):

```
s_i(z_i) = a (z_i)^α                                          (4)
```

and its Section IV extension, showing the rate also depends on the **local
material hold-up** `H` (paper eq. (16)):

```
s_i ∝ e^(-k_s H)                                               (16)
```

The code combines both into a single vectorized expression, per compartment
(using `a_cell[k]`) and per cell's *hold-up density* `H_density = H_cell/dx`
(tons per metre, matching the paper's `H` units of ~1–3 t/m from Table values):

```python
s[k, :] = a_cell[k] * (z ** alpha) * exp(-k_s * H_density[k])
```

This is the key nonlinearity #1 that the module's docstring calls out: higher
local hold-up chokes the breakage rate (a fuller mill grinds less efficiently
per unit time), which is the mechanism behind the mill's tendency to "flood"
if fed too fast, and the bell-shaped steady-state `M_P(H)` curve of paper
Fig. 8.

The finest class (index N−1) cannot break further; this is masked out via
`self.breakable`.

### Breakage distribution `b_{ij}` (paper eq. (5), (15), (2)–(3))

The paper's simplified cumulative breakage distribution (eq. (5), analogous
derivation to eq. (15) for the two-compartment case):

```
B_ij = (z_{i-1} / z_j)^β                                       (5)
```

`B_ij` is the *cumulative* fraction of debris from parent size `z_j` that ends
up finer than boundary `z_{i-1}`. The code's `_breakage_matrix()` converts this
cumulative distribution into the incremental `b_{ij}` used in eq. (6)
(paper eq. (2): `b_{ij} = B_{ij} - B_{i+1,j}`), i.e. the fraction of class-j
debris landing specifically in class `i`:

```python
cdf_i = (z[i] / z[j]) ** beta      # B evaluated at boundary z[i]
b[i, j] = prev_cdf - cdf_i          # incremental probability mass in class i
```

with the finest class acting as an absorbing bin (`cdf = 0` there, so it
catches everything not yet assigned), and each column renormalized to sum to
1 (mass conservation: everything broken out of class `j` must reappear
somewhere finer).

The breakage source term in `rhs()` implements the sum in eq. (6),
`Σ_{j<i} b_{ij} s_j H m_j`, vectorized as a matrix product over all cells:

```python
breakage_source = np.einsum('ij,kj->ki', b, s * W)   # Σ_j b[i,j] * (s*W)[k,j]
```

---

## 5. Air separator (paper Section II-C, eq. (10))

The paper models the separator via a **reduced efficiency curve**, a function
of particle size normalized by the separator cut point `z50c`:

```
E(z_r) = M_P(z_r) / M_M(z_r) = C e^(-D z_r) / (A z_r² - B z_r + 1),   z_r = z / z50c    (10)
```

an S-shaped curve from ~0 (coarse particles, mostly rejected) to 1 (fine
particles, mostly pass to product), with `E(1) = 50%` at the cut point by
definition, calibrated per register position and mill flow rate
(`A,B,C,D = A,B,C,D(P_reg, M_M)`, per Section III-B).

Note that `A` is not independent: the definition `E(1) = 50%` fixes it as
`A = 2C·e^(-D) + B - 1`, so eq. (10) has only three free shape parameters
(`B`, `C`, `D`) plus the cut point.

The code does **not** use the exact rational-times-exponential form of eq.
(10) directly (its A–D coefficients as functions of register/flow are not
published, being deferred to ref. [9], Boulvin's 2001 PhD dissertation).
Instead `separator_efficiency()` uses a **log-logistic** (Tromp-type)
partition curve with the same qualitative S-shape, guaranteed to stay
within `[f_min, f_max] = [0.03, 0.92]` (residual coarse escape / fine
bypass floors) rather than the exact `[C·something, 1]` range of eq. (10):

```python
zr = z / z50c
E = f_min + (f_max - f_min) / (1 + zr**m)        # m = sep_sharpness
```

evaluated internally as `exp(m·ln zr)` with the same `clip(..., -50, 50)`
guard so extreme size classes cannot over- or underflow.

`m` is the partition exponent (default 3.0, settable via the `sep_sharpness`
constructor argument): higher `m` = sharper cut. Expressed as the usual
sharpness index `κ = z25/z75` of the bypass-corrected curve,
`κ = 3^(-2/m)`, so `m = 3` gives `κ ≈ 0.48` — a realistic value for a
static/register-controlled separator (published high-efficiency classifiers
run 0.5–0.7).

This is a *logistic in log(z_r)* rather than in `z_r` itself, which matters
in two ways:

- **The bypass constant means what it says.** As `z_r → 0`, `zr**m → 0` and
  `E → f_max` exactly, so the fine-end plateau — and hence the separator
  bypass `1 - f_max = 8%` — is fixed by the constant alone. In the earlier
  `exp(σ(z_r - 1))` form `f_max` was an asymptote at `z_r → -∞`, unreachable
  for any physical size, so the *effective* bypass was ~12% and drifted with
  the sharpness (a hidden coupling between two nominally independent curve
  properties).
- **It is symmetric on the log-size axis**, which is how separation curves
  are measured and plotted (paper Figs. 2, 4, 5, all on a log size axis).

The log-logistic is a standard published partition form (the Plitt /
Lynch–Rao / logistic family used for classifiers and hydrocyclones). It
remains a *stand-in*, not eq. (10): being monotone, it cannot reproduce the
"fish-hook" — the local minimum near 0.03 mm visible in the paper's Fig. 2,
which eq. (10) generates through the minimum of its denominator at
`z_r = B/(2A)`. It is also better conditioned than eq. (10), whose
denominator can reach zero and blow the curve up unless `B² < 4A`.

Industrial context for the constants: Altun & Benzer, *Powder Technology*
264 (2014) 1–8, report `d50c = 0.03–0.11 mm` for cement classifiers
(bracketing `z50c_base = 0.09`), but bypass varying over **2–40%** with
feed dust loading — so the fixed `1 - f_max = 8%` here is at the optimistic
end and is the main remaining fidelity gap in this part of the model.

### Cut-point dependence on mill flow rate (nonlinearity #2)

The paper's central nonlinearity #2 (Section IV): the separator cut-point
`z50c` grows as the mill flow rate `M_M` falls (paper Fig. 5), which is "the
mechanism responsible for the closed-loop instability" per the module
docstring. Code:

```python
def separator_cutpoint(self, Mm):
    return z50c_base * sep_bias * (Mm_ref / Mm) ** p_cutpoint
```

`sep_bias` is the live register-position control (>1 = registers more
closed → coarser cut point → more material rejected to `M_R`; <1 = more open
→ finer cut, less recirculation) — the manual control path corresponding to
the paper's `P_reg`.

### Rotor-speed control path

As an alternative to a static/register-controlled separator, the code also
offers `set_rotor_speed(rpm)` for **dynamic** (high-efficiency, rotor-speed
controlled) separators — not covered by the 2003 paper, which used a static
separator. This is a simple force-balance approximation
(centrifugal force `∝ rpm² · mass`, balanced against aerodynamic drag,
giving cut size `∝ 1/rpm` to first order):

```python
ratio = nominal_rpm / rpm
sep_bias = ratio ** speed_cutpoint_exponent
# sep_sharpness = ... ** speed_sharpness_exponent   # disabled, see below
```

Both control paths act through the same `sep_bias` multiplier on `z50c`, so
they should be used interchangeably, not simultaneously.

The second line — rotor speed also sharpening the separation curve — is
**currently commented out**, so `sep_sharpness` is constant for the whole
simulation and `speed_sharpness_exponent` is unused. Two reasons to leave
it that way: it was a second-order effect (`m` moved only ~2.7→3.4 across a
wide speed range), and in the previous `exp(σ(z_r-1))` curve it silently
perturbed the bypass as well as the sharpness. The separator literature
also ties sharpness to classifier *load* rather than to rotor speed —
Altun & Benzer (2014) find sharpness *decreasing* with feed dust loading —
so if a variable sharpness is wanted, `M_M` is the physically supported
driving variable, with the opposite sign to the disabled coupling.

---

## 6. Overflow/relief transport response (not in the paper)

To keep hold-up bounded under sustained overfeeding — a purely numerical/
practical safeguard, not part of the published model — `_effective_u_cell()`
multiplies the nominal transport velocity by a smooth relief factor once a
compartment's hold-up passes 85% of a capacity limit `H_cap`:

```python
excess = max(0, (H - 0.85*H_cap) / (0.15*H_cap))
relief_multiplier = 1 + overflow_gain * excess**3
```

This mimics faster discharge through a mill's overflow trunnion / discharge
diaphragm as material level approaches the grate limit, keeping the
simulation numerically well-behaved (settling at/near `H_cap`) instead of
diverging under extreme feed steps — the paper instead notes (Section IV)
that in such extreme regimes "the measurements become unreliable" on the real
plant.

---

## 7. Fresh feed and its particle-size distribution

```python
w = np.linspace(1.0, 0.05, N)
mC = w / w.sum()
```

A simple triangular-ish profile weighted toward the coarsest classes,
reflecting that fresh clinker feed is coarse (the paper does not publish an
explicit `m_{F,i}`, only that it is the "feed mass fraction of size i" in
boundary condition eq. (8); the code supplies a physically reasonable
monotone-decreasing proxy).

---

## 8. Integration

Two integration entry points wrap `rhs()` (the discretized right-hand side of
eq. (6), a system of `Nx × N` coupled ODEs after spatial discretization —
the paper's own "method of lines" approach):

- `simulate(t_span, Mc_func, ...)`: full trajectory via `solve_ivp` with the
  stiff `BDF` method (breakage/transport time constants can differ by orders
  of magnitude, especially with the hold-up-dependent nonlinearity of eq.
  (16), so an implicit stiff solver is used for accuracy over long spans).
- `step(y0, dt, Mc)`: single-step advance via explicit `RK45`, tuned for
  cheap repeated calls (e.g. one animation frame in a real-time dashboard or
  one MPC prediction/control step) at constant feed rate.

---

## 9. Derived / output quantities

Both `instantaneous_outputs(y)` (from a raw state vector) and
`derived_outputs(sol)` (post-processing a full `solve_ivp` trajectory)
compute, per paper Fig. 6–8's plotted quantities:

- **`H1`, `H2`, `H`** — compartment and total hold-up (tons), via
  `compartment_holdups()`.
- **`M_M`** — mill product flow rate (t/h): the advective outflow of the last
  axial cell, `Mm_i = (u_last/dx) · W_last`, summed over classes and
  converted from t/min to t/h (`× 60`). This must use the *same* flux
  expression as the last cell's outflow term in `rhs()`, or the mass balance
  would not close (noted explicitly in the code).
- **`M_P`, `M_R`** — separator split of `M_M` via `E(Mm)`:
  `Mp_i = Mm_i · E`, `Mr_i = Mm_i · (1-E)`, corresponding to paper eq. (10)
  (`M_P = E · M_M`) and its underflow complement `M_R = (1-E) · M_M`
  (paper's Fig. 2, "underflow separator efficiency `1 - E(z_r) = M_R(z_r)/M_M(z_r)`").
- **Fineness proxy (`blaine_raw`)** — a simplified stand-in for the paper's
  measured specific surface (cm²/g, its Fig. 6 bottom-right plot), computed
  as a mass-weighted sum of `1/z` over the product stream's size
  distribution, scaled by a calibration constant `_blaine_cal`:

  ```python
  mp_frac = Mp_i / Mp_i.sum()
  blaine_raw = _blaine_cal * Σ(mp_frac / z)
  ```

  finer product (smaller `z`, more mass in fine classes) → higher proxy
  value, qualitatively tracking specific surface without being a
  first-principles surface-area calculation.

### Output-range shaping

Because the raw fineness and recirculation numbers can swing very widely in
extreme (e.g. flooding) regimes — not a concern the 2003 paper needed to
address for a plotting/dashboard context — both are passed through smooth
saturating maps before being reported, so the dashboard/plots stay in a
realistic range without discarding information via hard clipping:

- **Blaine**: logistic map into `[BLAINE_MIN, BLAINE_MAX] = [2200, 7200]` cm²/g,
  centered at `BLAINE_CENTER = 4500` with scale `BLAINE_SCALE = 900`.
- **Recirculated flow `M_R`**: exponential-saturation map into
  `[0, MR_MAX] = [0, 300]` t/h with time-constant-like parameter
  `MR_TAU = 300`.

Both maps are near-identity in the normal operating region and only
saturate far from it.

---

## 10. Summary: code ↔ paper equation map

| Code element | Paper reference |
|---|---|
| `self.z` geometric size grid | Sec. II-A, text after eq. (3): `R = z_j/z_{j+1}` |
| `s[k,:] = a_cell*z**alpha*exp(-k_s*H_density)` | eq. (4) `s_i=a z_i^α`, combined with eq. (16) `s_i ∝ e^{-k_s H}` |
| `_breakage_matrix()` → `self.b` | eq. (5) `B_ij=(z_{i-1}/z_j)^β`, eq. (2) `b_{ij}=B_{ij}-B_{i+1,j}` |
| `rhs()`: `conv + diff - breakage_sink + breakage_source` | eq. (6), the governing PDE, discretized (method of lines) |
| feed injection at cell 0 | eq. (8), boundary condition at `x=0` |
| outflow from last cell | eq. (9), boundary condition at `x=L` (with `Pr_i = 0`, i.e. no grate rejection) |
| `separator_efficiency()` | eq. (10), reduced efficiency curve `E(z_r)` (log-logistic stand-in, not the exact rational form; no fish-hook) |
| `separator_cutpoint()` mill-flow dependence | Sec. IV nonlinearity #2, Fig. 5 |
| two-compartment `a_cell`, `u_cell_base` split | Sec. II, Table I/II compartment-specific parameters |
| `Mp = Mm*E`, `Mr = Mm*(1-E)` | eq. (10) and its underflow complement (Fig. 2) |
| `blaine_raw` fineness proxy | Fig. 6 specific-surface measurement (simplified proxy, not first-principles) |
| `_effective_u_cell()` overflow relief | not in the paper — numerical safeguard against unbounded hold-up |
| `set_rotor_speed()` | not in the paper — added dynamic-separator control path |
| output-range shaping (`_map_blaine`, `_map_return_flow`) | not in the paper — dashboard/plotting convenience |

The unmodeled/simplified aspects are all explicitly acknowledged in the
module's own docstring (lines 22–30 of `cement_mill_sim.py`): exact axial
diffusion coefficient, per-compartment velocities, and full separator A–D
constants are not published in the paper, so representative values of the
right order of magnitude are used, preserving the paper's *qualitative*
dynamics (bell-shaped `M_P(H)`, mill-emptying under negative feed steps,
flow-rate-sensitive separator cut point) rather than exact numerical
reproduction.
