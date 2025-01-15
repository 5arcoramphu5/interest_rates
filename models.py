import numpy as np
import sympy as sym
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
        
        def b(x: sym.Float, t: sym.Float) -> sym.Float:
            return self.sigma
        
        return SDE(a, b)
    
class CIRModel(Model):
    
    def __init__(self, theta: float, alpha: float, sigma: float):
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma

    def getSDE(self) -> SDE:
        def a(x: float, t: float) -> float:
            return self.theta - self.alpha * x
        
        def b(x: sym.Float, t: sym.Float) -> sym.Float:
            return self.sigma * sym.sqrt(x)
        
        return SDE(a, b)
    

class RBModel(Model):
    
    def __init__(self, theta: float, sigma: float):
        self.theta = theta
        self.sigma = sigma

    def getSDE(self) -> SDE:
        def a(x: float, t: float) -> float:
            return self.theta * x
        
        def b(x: sym.Float, t: sym.Float) -> sym.Float:
            return self.sigma * x
        
        return SDE(a, b)