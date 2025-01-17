import numpy as np
import sympy as sym
from abc import ABC, abstractmethod
from SDE_schemes import SDE
from pandas import Series

class InterestRateModel(ABC):
    """
    Abstract class for defining a interest rate model.
    """

    @abstractmethod
    def getSDE() -> SDE:
        """
        A stochastic differential equation of the form dX_t = a(X_t, t)dt + b(X_t, t)dW_t.
        """
        pass

    @abstractmethod
    def calibrate(self, rates: Series, dt: float) -> float:
        """
        Calibrates the model to market data. Returns r0.
        """
        pass


class VasicekModel(InterestRateModel):
    """
    Vasicek Model for interest rate modeling.
    The Vasicek model is a type of one-factor short rate model that describes the evolution of interest rates. 

    where:
    - alpha: speed of reversion to the mean
    - theta: long-term mean level of the interest rate
    - sigma: volatility of the interest rate
    - r0: initial interest rate
    Methods:
        getSDE() -> SDE:
            Returns the stochastic differential equation (SDE) for the Vasicek model.

        calibrate(rates, dt)
            Calibrates the model parameters based on historical interest rate data.
            Args:
                rates (Series): Historical interest rate data.
                dt (float): Time increment.
    """
    
    def __init__(self, alpha: float = 0, theta: float = 0, sigma: float = 0):
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma

    def getSDE(self) -> SDE:
        def a(x: float, t: float) -> float:
            return self.theta - self.alpha * x
        
        def b(x: sym.Float, t: sym.Float) -> sym.Float:
            return self.sigma
        
        return SDE(a, b)
    
    def calibrate(self, rates: Series, dt:float) -> None:
        N = len(rates)
        
        Sx = sum(rates.iloc[0:(N-1)])
        Sy = sum(rates.iloc[1:N])
        Sxx = np.dot(rates.iloc[0:(N-1)], rates.iloc[0:(N-1)])
        Sxy = np.dot(rates.iloc[0:(N-1)], rates.iloc[1:N])
        Syy = np.dot(rates.iloc[1:N], rates.iloc[1:N])

        theta = (Sy * Sxx - Sx * Sxy) / (N * (Sxx - Sxy) - (Sx**2 - Sx*Sy))
        kappa = -np.log((Sxy - theta * Sx - theta * Sy + N * theta**2) / (Sxx - 2*theta*Sx + N*theta**2)) / dt
        a = np.exp(-kappa * dt)
        sigmah2 = (Syy - 2*a*Sxy + a**2 * Sxx - 2*theta*(1-a)*(Sy - a*Sx) + N*theta**2 * (1-a)**2) / N
        self.sigma = np.sqrt(sigmah2*2*kappa / (1-a**2))
        self.theta = theta * kappa
        self.alpha = kappa

class CIRModel(InterestRateModel):
    
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
    
    def calibrate(self, rates: Series, dt:float) -> None:
        pass