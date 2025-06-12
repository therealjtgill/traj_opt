import copy
import numpy as np
from typing import List

from cartpole_dynamics import CartpoleDynamics
from quadratic_program import QuadraticProgram
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
      self.dynamics = dynamics
      self.num_collocation_points = num_collocation_points
      self.time_horizon = time_horizon
      self.initial_state = initial_state
      self.final_state = final_state
      self.reference_trajectory = reference_trajectory
      self.relinearization_sequence = relinearization_sequence

      self.qp = QuadraticProgram()
