import abc
import torch

class GWOModule(torch.nn.Module, abc.ABC):
    """
    Abstract base class for all benchmarked models.
    Inherits from torch.nn.Module and abc.ABC.
    """

    @property
    @abc.abstractmethod
    def C_D(self) -> int:
        """
        Descriptive Complexity (C_D): The number of 'primitive' operations
        required to describe the module's structure. This must be calculated
        and provided by the user.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_parametric_complexity_modules(self) -> list[torch.nn.Module]:
        """
        Returns a list of torch.nn.Module instances whose parameters should be
        counted towards the Parametric Complexity (C_P). If no modules are
        relevant for C_P calculation, this should return an empty list.
        """
        raise NotImplementedError
