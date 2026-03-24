"""
Hybrid Physics-ML Digital Twin

Combines the RC thermal network (physics) with the neural correction (ML)
into a single prediction pipeline:

    T_predicted = RC_model(inputs) + NN_correction(features)

This is the same hybrid architecture pattern from CSH2's digital twin,
applied to a different physical domain.
"""

import numpy as np
import torch
from typing import Optional

from src.physics.rc_model import RC3R2CModel, RC3R2CParams, ZoneInputs
from src.ml.correction import ResidualCorrectionNet


class BuildingDigitalTwin:
    """Complete hybrid digital twin for a building thermal zone.

    Usage:
        1. Calibrate the RC model on historical data
        2. Train the correction network on RC model residuals
        3. Predict: T = RC_model + NN_correction
    """

    def __init__(
        self,
        rc_params: Optional[RC3R2CParams] = None,
        correction_model: Optional[ResidualCorrectionNet] = None,
    ):
        self.rc_model = RC3R2CModel(rc_params)
        self.correction_model = correction_model
        self.physics_weight = 1.0  # Blend factor (1.0 = trust physics fully)

    def predict_physics_only(
        self,
        inputs: ZoneInputs,
        T_wall_0: float = 20.0,
        T_int_0: float = 22.0,
    ) -> np.ndarray:
        """Physics-only prediction (RC model)."""
        result = self.rc_model.simulate(inputs, T_wall_0, T_int_0)
        return result["T_int"]

    def predict_hybrid(
        self,
        inputs: ZoneInputs,
        correction_features: torch.Tensor,
        T_wall_0: float = 20.0,
        T_int_0: float = 22.0,
    ) -> np.ndarray:
        """Hybrid prediction: physics + ML correction."""
        T_physics = self.predict_physics_only(inputs, T_wall_0, T_int_0)

        if self.correction_model is not None:
            self.correction_model.eval()
            with torch.no_grad():
                correction = self.correction_model(correction_features).numpy()
            T_hybrid = T_physics + correction
        else:
            T_hybrid = T_physics

        return T_hybrid

    def evaluate(
        self,
        T_predicted: np.ndarray,
        T_measured: np.ndarray,
    ) -> dict[str, float]:
        """Compute evaluation metrics."""
        residual = T_predicted - T_measured
        return {
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "max_error": float(np.max(np.abs(residual))),
            "r2": float(1 - np.sum(residual ** 2) / np.sum(
                (T_measured - T_measured.mean()) ** 2
            )),
            "bias": float(np.mean(residual)),
        }
