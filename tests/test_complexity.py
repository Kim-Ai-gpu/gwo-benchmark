import unittest
import torch
from gwo_benchmark.base import GWOModule
from gwo_benchmark.complexity import ComplexityCalculator

class MockGWOModule(GWOModule):
    def __init__(self, c_d, param_modules):
        super().__init__()
        self._c_d = c_d
        self._param_modules = torch.nn.ModuleList(param_modules)

    @property
    def C_D(self) -> int:
        return self._c_d

    def get_parametric_complexity_modules(self) -> list[torch.nn.Module]:
        return list(self._param_modules)

    def forward(self, x):
        pass

class TestComplexityCalculator(unittest.TestCase):

    def test_calculation(self):
        """Tests the basic complexity calculation."""
        param_modules = [
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 2)
        ]
        module = MockGWOModule(c_d=10, param_modules=param_modules)

        calculator = ComplexityCalculator(alpha=1.0)
        results = calculator.calculate(module)

        self.assertEqual(results["c_d"], 10)

        self.assertAlmostEqual(results["c_p_M"], 0.000067)

        self.assertAlmostEqual(results["omega_proxy"], 10.000067)

    def test_no_parametric_complexity(self):
        """Tests calculation when there are no parametric complexity modules."""
        module = MockGWOModule(c_d=5, param_modules=[])

        calculator = ComplexityCalculator(alpha=1.0)
        results = calculator.calculate(module)

        self.assertEqual(results["c_d"], 5)
        self.assertEqual(results["c_p_M"], 0.0)
        self.assertEqual(results["omega_proxy"], 5.0)

if __name__ == '__main__':
    unittest.main()
