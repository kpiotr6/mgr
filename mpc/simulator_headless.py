"""Headless cement mill simulator.

This is a lightweight synthetic simulator used for MPC loop testing.

Inputs (controls):
- clinker_1_feedrate: [0, 120]
- separator_speed: [280, 800]

Outputs (process variables):
- return
- first_chamber_filling
- second_chamber_filling
- gran1_blain
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


@dataclass
class CementMillSimConfig:
    k1: float = 0.5
    k2: float = 0.3
    dt: float = 1.0

    # Max holdup per compartment (tonnes). Fillings are reported as % occupied.
    compartment_capacity_tonnes: float = 200.0

    # Noise
    holdup_noise_std: float = 1.2
    return_noise_std: float = 0.6
    blain_noise_std: float = 15.0

    # Control bounds
    clinker_1_feedrate_min: float = 2.0
    clinker_1_feedrate_max: float = 120.0
    separator_speed_min: float = 280.0
    separator_speed_max: float = 800.0


class CementMillSim:
    def __init__(self, config: CementMillSimConfig | None = None, seed: int | None = 0):
        self.config = config or CementMillSimConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, *, H1: float = 75.0, H2: float = 75.0, time: float = 0.0) -> None:
        cap = float(self.config.compartment_capacity_tonnes)
        self.H1 = _clip(float(H1), 0.0, cap)
        self.H2 = _clip(float(H2), 0.0, cap)
        self.time = float(time)

    def step(self, *, clinker_1_feedrate: float, separator_speed: float) -> dict[str, float]:
        cfg = self.config
        cap = float(cfg.compartment_capacity_tonnes)

        F = _clip(clinker_1_feedrate, cfg.clinker_1_feedrate_min, cfg.clinker_1_feedrate_max)
        S = _clip(separator_speed, cfg.separator_speed_min, cfg.separator_speed_max)

        # Map separator speed (280..800) -> (0..100) internal %
        S_pct = (S - cfg.separator_speed_min) / (cfg.separator_speed_max - cfg.separator_speed_min) * 100.0

        alpha = 0.2 + (S_pct / 100.0) * 0.6

        P1 = cfg.k1 * self.H1
        P2 = cfg.k2 * self.H2
        R_clean = alpha * P2

        dH1 = F + R_clean - P1
        dH2 = P1 - P2

        self.H1 += dH1 * cfg.dt
        self.H2 += dH2 * cfg.dt
        self.H1 = _clip(self.H1, 0.0, cap)
        self.H2 = _clip(self.H2, 0.0, cap)
        self.time += cfg.dt

        # Blaine (synthetic relationship)
        B_clean = 2200.0 + (S_pct * 15.0) + ((self.H1 + self.H2) / (F + 1.0) * 25.0)

        # Noisy measurements
        H1_noisy = _clip(self.H1 + self.rng.normal(0.0, cfg.holdup_noise_std), 0.0, cap)
        H2_noisy = _clip(self.H2 + self.rng.normal(0.0, cfg.holdup_noise_std), 0.0, cap)
        R_noisy = max(0.0, R_clean + self.rng.normal(0.0, cfg.return_noise_std))
        B_noisy = B_clean + self.rng.normal(0.0, cfg.blain_noise_std)

        first_chamber_filling = _clip((H1_noisy / cap) * 100.0, 0.0, 100.0)
        second_chamber_filling = _clip((H2_noisy / cap) * 100.0, 0.0, 100.0)

        return {
            "return": float(R_noisy),
            "first_chamber_filling": float(first_chamber_filling),
            "second_chamber_filling": float(second_chamber_filling),
            "gran1_blain": float(B_noisy),
            "time": float(self.time),
        }


if __name__ == "__main__":
    # Tiny smoke demo: run 200 steps with constant inputs.
    sim = CementMillSim(seed=0)
    for _ in range(5):
        y = sim.step(clinker_1_feedrate=60.0, separator_speed=540.0)
        print(y)