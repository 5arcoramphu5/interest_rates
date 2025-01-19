import numpy as np
import sympy as sym
from abc import ABC, abstractmethod
from SDE_schemes import SDE, EulerMaruyamaSolver
from pandas import Series
from sklearn.linear_model import LinearRegression
from collections.abc import Callable
from scipy.optimize import minimize

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
    def calibrate(self, rates: Series, dt: float, t0: float, tn: float) -> float:
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
    
    def calibrate(self, rates: Series, dt:float, t0: float, tn: float) -> None:
        N = len(rates)
        
        np_rates = rates.to_numpy(dtype=np.float64)
        x = np_rates[:-1]
        y = np_rates[1:]
        Sx = sum(x)
        Sy = sum(y)
        Sxx = np.dot(x, x)
        Sxy = np.dot(x, y)
        Syy = np.dot(y, y)

        theta = (Sy * Sxx - Sx * Sxy) / (N * (Sxx - Sxy) - (Sx**2 - Sx*Sy))
        kappa = -np.log((Sxy - theta * Sx - theta * Sy + N * theta**2) / (Sxx - 2*theta*Sx + N*theta**2)) / dt
        a = np.exp(-kappa * dt)
        sigmah2 = (Syy - 2*a*Sxy + a**2 * Sxx - 2*theta*(1-a)*(Sy - a*Sx) + N*theta**2 * (1-a)**2) / N
        self.sigma = np.sqrt(sigmah2*2*kappa / (1-a**2))
        self.theta = theta * kappa
        self.alpha = kappa

class CIRModel(InterestRateModel):
    
    def __init__(self, theta: float = 0, alpha: float = 0, sigma: float = 0):
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma

    def getSDE(self) -> SDE:
        def a(x: float, t: float) -> float:
            return self.theta - self.alpha * x
        
        def b(x: sym.Float, t: sym.Float) -> sym.Float:
            return self.sigma * sym.sqrt(x)
        
        return SDE(a, b)
    
    def calibrate(self, rates: Series, dt:float, t0: float, tn: float) -> None:

        np_rates = rates.to_numpy(dtype=np.float64)
        rs = np_rates[:-1]
        rt = np_rates[1:]

        rs_sqrt = np.sqrt(rs)
        y = (rt - rs) / rs_sqrt
        X = np.column_stack((dt / rs_sqrt, dt * rs_sqrt))

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        prediction = model.predict(X)
        residuals = y - prediction
        beta1 = model.coef_[0]        
        beta2 = model.coef_[1]

        k0 = -beta2
        theta0 = beta1/k0
        sigma0 = np.std(residuals) / np.sqrt(dt)

        self.theta = k0*theta0
        self.alpha = k0
        self.sigma = sigma0


class HWModel(InterestRateModel):
    
    def __init__(self, theta: np.array = [], alpha: float = 0, sigma: np.array = [], timeToIndex: Callable[[float], int] = lambda f: int(f)):
        self.theta = theta
        self.alpha = alpha
        self.sigma = sigma
        self.timeToIndex = timeToIndex
        self.N = len(self.theta)

    def getSDE(self) -> SDE:
        def a(x: float, t: float) -> float:
            tInt = self.timeToIndex(t)
            return self.theta[tInt] - self.alpha * x
        
        def b(x: sym.Float, t: sym.Float) -> sym.Float:
            tInt = self.timeToIndex(t)
            return self.sigma[tInt]
        
        return SDE(a, b, lambda x, t: 0)
    
    def calibrate(self, rates: Series, dt:float, t0: float, tn: float) -> None:

        N = len(rates)
        timeToIndex = lambda t: round((t - t0) / dt)
        seed = 1

        def calibration_obj(x, r0, np_rates, t0, tn, N):
            theta, sigma, alpha = x[:N], x[N:2*N], x[2*N:][0]

            model = HWModel(theta, alpha, sigma, timeToIndex)
            t, model_result = EulerMaruyamaSolver.performSimulation(model.getSDE(), r0, t0, tn, N-1, seed)

            return np.sum((model_result - np_rates[:-1])**2)

        np_rates = rates.to_numpy(dtype=np.float64)
        theta0 = np.ones(N) * 0.05
        sigma0 = np.ones(N) * 0.01
        alpha0 = 0.3
        x0 = np.concatenate([theta0, sigma0, [alpha0] ])

        result = minimize(calibration_obj, x0, args=(np_rates[0], np_rates, t0, tn, N), options={'maxiter': 100})
        print("success:", result.success, ", message:", result.message)
        self.theta = result.x[:N]
        self.sigma = result.x[N:2*N]
        self.alpha = result.x[2*N:][0]