import numpy as np
from abc import ABC, abstractmethod
from SDE_schemes import SDE

class Model(ABC):
    """
    Abstract class for defining a interest rate model.
    """

    @abstractmethod
    def getSDE() -> SDE:
        """
        A stochastic differential equation of the form dX_t = a(X_t, t)dt + b(X_t, t)dW_t.
        """
        pass

class VasicekModel(Model):
    
    def __init__(self, theta: float, alpha: float, sigma: float):
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma

    def getSDE(self) -> SDE:
        def a(x: float, t: float) -> float:
            return self.theta - self.alpha * x
        
        def b(x: float, t: float) -> float:
            return self.sigma
        
        def b_der(x: float, t: float) -> float:
            return 0
        
        return SDE(a, b, b_der)
    
class CIRModel(Model):
    
    def __init__(self, theta: float, alpha: float, sigma: float):
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma

    def getSDE(self) -> SDE:
        def a(x: float, t: float) -> float:
            return self.theta - self.alpha * x
        
        def b(x: float, t: float) -> float:
            return self.sigma * np.sqrt(x)
        
        def b_der(x: float, t: float) -> float:
            return self.sigma / (2 * np.sqrt(x))
        
        return SDE(a, b, b_der)