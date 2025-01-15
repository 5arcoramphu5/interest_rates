import numpy as np
import sympy as sym
from sympy.abc import x, t
from typing import Tuple
from collections.abc import Callable

class SDE:
    """
    Class for defining a stochastic differential equation of the form dX_t = a(X_t, t)dt + b(X_t, t)dW_t.
    """
    def __init__(self, a: Callable[[float, float], float], b: Callable[[sym.Float, sym.Float], sym.Float]):
        self.a = a
        self.b = lambda x, t: float(b(x, t))
        
        b_der_sym = sym.diff(b(x, t), x)
        self.b_der = sym.lambdify([x, t], b_der_sym)
        

def dW(dt: float):
    return np.random.normal(0, np.sqrt(dt))
    
def euler_maruyama(sde: SDE, y0: float, t0: float, tN: float, N: int) -> Tuple[np.array, np.array]:
    """
    Euler-Maruyama scheme for solving stochastic differential equations of form dX_t = a(X_t, t)dt + b(X_t, t)dW_t.

    Parameters:
    sde: SDE
        Object representing the SDE.
    x0: float
        The initial condition X_0 = x0.
    t0: float
        The initial time.
    tN: float
        The final time.
    N: int
        The number of time steps.

    Returns:
    t: numpy array
        The time grid.
    x: numpy array
        The solution of the SDE at the time grid points.
    """
    dt = (tN - t0) / N
    t = np.linspace(t0, tN, N+1)

    Y = np.zeros(N+1)
    Y[0] = y0

    for i in range(N):
        Y[i+1] = Y[i] + sde.a(Y[i], t[i]) * dt + sde.b(Y[i], t[i]) * dW(dt)

    return t, Y


def milstein(sde: SDE, y0: float, t0: float, tN: float, N: int) -> Tuple[np.array, np.array]:
    """
    Milstein scheme for solving stochastic differential equations of form dX_t = a(X_t)dt + b(X_t)dW_t.

    Parameters:
    sde: SDE
        Object representing the SDE.
    x0: float
        The initial condition X_0 = x0.
    t0: float
        The initial time.
    tN: float
        The final time.
    N: int
        The number of time steps.

    Returns:
    t: numpy array
        The time grid.
    x: numpy array
        The solution of the SDE at the time grid points.
    """
    dt = (tN - t0) / N
    t = np.linspace(t0, tN, N+1)

    Y = np.zeros(N+1)
    Y[0] = y0

    for i in range(N):
        currdW = dW(dt)
        Y[i+1] = Y[i] + sde.a(Y[i], t[i]) * dt + sde.b(Y[i], t[i]) * currdW + sde.b(Y[i], t[i]) * sde.b_der(Y[i], t[i]) / 2 * (currdW**2 - dt)

    return t, Y