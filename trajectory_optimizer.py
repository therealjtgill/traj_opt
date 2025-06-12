import numpy as np
from typing import List

from cartpole_dynamics import CartpoleDynamics
from solvers import FeasibleStartNewton, InfeasibleStartNewton

class TrajectoryOptimizer:
   def __init__(
      self,
      dynamics: CartpoleDynamics,
      num_collocation_points: int,
      time_horizon: float,
      initial_state: np.ndarray,
      final_state: np.ndarray = None,
      reference_trajectory: List[np.ndarray] = None,
      relinearization_sequence: List[np.ndarray] = None
   ):
      pass
