"""19-dimensional state representation with boxplus/boxminus on SO(3).

Matches C++ StatesGroup from include/common_lib.h:126-223.

State vector layout for boxplus/boxminus:
    [0:3]   rotation (SO(3), via Exp/Log maps)
    [3:6]   position
    [6]     inverse exposure time
    [7:10]  velocity
    [10:13] gyroscope bias
    [13:16] accelerometer bias
    [16:19] gravity vector
"""
import numpy as np
from .so3 import exp_so3, log_so3

DIM_STATE = 19
INIT_COV = 0.01
G_m_s2 = 9.81


def _init_cov() -> np.ndarray:
    """Initialize the state covariance matrix matching C++ StatesGroup constructor."""
    cov = np.eye(DIM_STATE) * INIT_COV
    cov[6, 6] = 0.00001
    cov[10:19, 10:19] = np.eye(9) * 0.00001
    return cov


class StatesGroup:
    """19-dimensional state for LIO EKF.

    Matches C++ StatesGroup exactly, including boxplus (operator+/+=)
    and boxminus (operator-) semantics.
    """

    __slots__ = ['rot_end', 'pos_end', 'vel_end', 'inv_expo_time',
                 'bias_g', 'bias_a', 'gravity', 'cov']

    def __init__(self):
        self.rot_end = np.eye(3)
        self.pos_end = np.zeros(3)
        self.vel_end = np.zeros(3)
        self.inv_expo_time = 1.0
        self.bias_g = np.zeros(3)
        self.bias_a = np.zeros(3)
        self.gravity = np.zeros(3)
        self.cov = _init_cov()

    def boxplus(self, delta: np.ndarray) -> 'StatesGroup':
        """state (+) delta -> new state. Matches C++ operator+.

        Args:
            delta: (19,) state increment vector

        Returns:
            New StatesGroup with the increment applied.
        """
        s = StatesGroup()
        s.rot_end = self.rot_end @ exp_so3(delta[0:3])
        s.pos_end = self.pos_end + delta[3:6]
        s.inv_expo_time = self.inv_expo_time + delta[6]
        s.vel_end = self.vel_end + delta[7:10]
        s.bias_g = self.bias_g + delta[10:13]
        s.bias_a = self.bias_a + delta[13:16]
        s.gravity = self.gravity + delta[16:19]
        s.cov = self.cov.copy()
        return s

    def boxplus_inplace(self, delta: np.ndarray):
        """In-place boxplus. Matches C++ operator+=.

        Args:
            delta: (19,) state increment vector
        """
        self.rot_end = self.rot_end @ exp_so3(delta[0:3])
        self.pos_end = self.pos_end + delta[3:6]
        self.inv_expo_time = self.inv_expo_time + delta[6]
        self.vel_end = self.vel_end + delta[7:10]
        self.bias_g = self.bias_g + delta[10:13]
        self.bias_a = self.bias_a + delta[13:16]
        self.gravity = self.gravity + delta[16:19]

    def boxminus(self, other: 'StatesGroup') -> np.ndarray:
        """self (-) other -> delta vector. Matches C++ operator-.

        Args:
            other: The state to subtract from self.

        Returns:
            (19,) difference vector in the tangent space.
        """
        delta = np.zeros(DIM_STATE)
        rotd = other.rot_end.T @ self.rot_end
        delta[0:3] = log_so3(rotd)
        delta[3:6] = self.pos_end - other.pos_end
        delta[6] = self.inv_expo_time - other.inv_expo_time
        delta[7:10] = self.vel_end - other.vel_end
        delta[10:13] = self.bias_g - other.bias_g
        delta[13:16] = self.bias_a - other.bias_a
        delta[16:19] = self.gravity - other.gravity
        return delta

    def copy(self) -> 'StatesGroup':
        """Deep copy of the state."""
        s = StatesGroup()
        s.rot_end = self.rot_end.copy()
        s.pos_end = self.pos_end.copy()
        s.vel_end = self.vel_end.copy()
        s.inv_expo_time = self.inv_expo_time
        s.bias_g = self.bias_g.copy()
        s.bias_a = self.bias_a.copy()
        s.gravity = self.gravity.copy()
        s.cov = self.cov.copy()
        return s

    def resetpose(self):
        """Reset pose and velocity to identity/zero. Matches C++ resetpose()."""
        self.rot_end = np.eye(3)
        self.pos_end = np.zeros(3)
        self.vel_end = np.zeros(3)
