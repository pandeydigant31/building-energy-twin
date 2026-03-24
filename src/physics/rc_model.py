"""
Resistance-Capacitance (RC) Thermal Network Model

Models a building thermal zone using the electrical analogy:
- Thermal resistance (R) ↔ Electrical resistance (insulation, convection)
- Thermal capacitance (C) ↔ Electrical capacitance (thermal mass)
- Temperature (T) ↔ Voltage
- Heat flow (Q) ↔ Current

The 3R2C model captures:
- R_wall: wall conduction resistance
- R_win: window/infiltration resistance
- R_int: internal-to-wall resistance
- C_wall: wall thermal mass
- C_int: interior air thermal mass

This is the physics backbone of the digital twin.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RC3R2CParams:
    """Parameters for a 3R2C thermal zone model.

    Default values are for a typical office zone (~200 m²).
    Units: R in K/W, C in J/K.
    """
    R_wall: float = 0.004    # Wall conduction resistance (K/W)
    R_win: float = 0.008     # Window + infiltration resistance (K/W)
    R_int: float = 0.002     # Interior-to-wall node resistance (K/W)
    C_wall: float = 5.0e6    # Wall thermal capacitance (J/K)
    C_int: float = 2.0e6     # Interior air thermal capacitance (J/K)


@dataclass
class ZoneInputs:
    """Time-varying inputs to the thermal zone model.

    All arrays must have the same length (number of timesteps).
    """
    T_outside: np.ndarray          # Outside air temperature (°C)
    Q_solar: np.ndarray            # Solar heat gains (W)
    Q_internal: np.ndarray         # Internal gains: people + equipment + lights (W)
    Q_hvac: np.ndarray             # HVAC heating (+) or cooling (-) power (W)
    dt: float = 3600.0             # Timestep in seconds (default: 1 hour)


class RC3R2CModel:
    """3R2C lumped-parameter thermal zone model.

    State variables:
        T_wall: wall node temperature (°C)
        T_int: interior air temperature (°C)

    The ODE system:
        C_wall * dT_wall/dt = (T_out - T_wall)/R_wall - (T_wall - T_int)/R_int
        C_int * dT_int/dt  = (T_wall - T_int)/R_int + (T_out - T_int)/R_win
                             + Q_solar + Q_internal + Q_hvac
    """

    def __init__(self, params: Optional[RC3R2CParams] = None):
        self.params = params or RC3R2CParams()

    def _ode(
        self, t: float, state: np.ndarray,
        T_out: float, Q_solar: float, Q_int: float, Q_hvac: float,
    ) -> np.ndarray:
        """ODE right-hand side for the 3R2C model."""
        T_wall, T_int = state
        p = self.params

        dT_wall = (
            (T_out - T_wall) / p.R_wall - (T_wall - T_int) / p.R_int
        ) / p.C_wall

        dT_int = (
            (T_wall - T_int) / p.R_int
            + (T_out - T_int) / p.R_win
            + Q_solar + Q_int + Q_hvac
        ) / p.C_int

        return np.array([dT_wall, dT_int])

    def simulate(
        self,
        inputs: ZoneInputs,
        T_wall_0: float = 20.0,
        T_int_0: float = 22.0,
    ) -> dict[str, np.ndarray]:
        """Run the RC model forward in time.

        Uses Euler integration for speed (sufficient at hourly resolution).

        Returns:
            Dict with 'T_wall', 'T_int' arrays (length = len(inputs.T_outside))
        """
        n_steps = len(inputs.T_outside)
        T_wall = np.zeros(n_steps)
        T_int = np.zeros(n_steps)
        T_wall[0] = T_wall_0
        T_int[0] = T_int_0

        for i in range(1, n_steps):
            state = np.array([T_wall[i - 1], T_int[i - 1]])
            deriv = self._ode(
                0, state,
                inputs.T_outside[i - 1],
                inputs.Q_solar[i - 1],
                inputs.Q_internal[i - 1],
                inputs.Q_hvac[i - 1],
            )
            new_state = state + deriv * inputs.dt
            T_wall[i] = new_state[0]
            T_int[i] = new_state[1]

        return {"T_wall": T_wall, "T_int": T_int}

    def calibrate(
        self,
        inputs: ZoneInputs,
        T_measured: np.ndarray,
        param_bounds: Optional[dict] = None,
    ) -> RC3R2CParams:
        """Calibrate RC parameters to match measured indoor temperature.

        Uses scipy.optimize.minimize (L-BFGS-B) to minimize
        RMSE between simulated T_int and measured temperature.
        """
        from scipy.optimize import minimize

        default_bounds = {
            "R_wall": (0.001, 0.02),
            "R_win": (0.002, 0.05),
            "R_int": (0.0005, 0.01),
            "C_wall": (1e5, 5e7),
            "C_int": (5e4, 1e7),
        }
        bounds = param_bounds or default_bounds

        param_names = list(bounds.keys())
        x0 = np.array([getattr(self.params, n) for n in param_names])
        b = [bounds[n] for n in param_names]

        def objective(x):
            for name, val in zip(param_names, x):
                setattr(self.params, name, val)
            result = self.simulate(inputs)
            rmse = np.sqrt(np.mean((result["T_int"] - T_measured) ** 2))
            return rmse

        res = minimize(objective, x0, method="L-BFGS-B", bounds=b)

        # Apply best params
        for name, val in zip(param_names, res.x):
            setattr(self.params, name, val)

        return self.params
