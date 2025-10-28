from .base import GWOModule

class ComplexityCalculator:
    """Calculates the complexity of a GWOModule."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def calculate(self, model: GWOModule) -> dict:
        """Calculates C_D, C_P, and Omega_proxy."""
        c_d = model.C_D
        
        parametric_modules = model.get_parametric_complexity_modules()
        c_p = sum(p.numel() for m in parametric_modules for p in m.parameters() if p.requires_grad)
        c_p_M = c_p / 1_000_000

        omega_proxy = c_d + self.alpha * c_p_M

        return {
            "c_d": c_d,
            "c_p_M": c_p_M,
            "omega_proxy": omega_proxy,
        }
