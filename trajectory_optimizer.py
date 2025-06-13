import copy
import numpy as np
from typing import List, Tuple

from cartpole_dynamics import CartpoleDynamics
from dynamics import phi_and_phi_integ
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
      reference_trajectory: List[Tuple[np.ndarray, np.ndarray]] | None = None, # (state, input)
      relinearization_sequence: List[np.ndarray] | None = None
   ):
      assert(num_collocation_points > 0)
      assert(time_horizon > 0.0)
      assert(relinearization_sequence is None or len(relinearization_sequence) == num_collocation_points)

      self.dynamics: CartpoleDynamics = dynamics
      self.num_collocation_points: int = num_collocation_points
      self.time_horizon: float = time_horizon
      self.initial_state: np.ndarray = initial_state
      self.final_state: np.ndarray | None = final_state
      self.reference_trajectory: List[Tuple[np.ndarray, np.ndarray]] | None = reference_trajectory
      self.relinearization_sequence: List[np.ndarray] | None = relinearization_sequence

      self.state_size: int = self.dynamics.state_size()
      self.control_size: int = self.dynamics.control_size()
      self.decision_variable_size = self.num_collocation_points * (self.state_size + self.control_size)

      # decision variable layout is
      # [control inputs, states]

      Q = np.zeros((self.decision_variable_size, self.decision_variable_size))
      Q[
         0: self.control_size * self.num_collocation_points,
         0: self.control_size * self.num_collocation_points
      ] = np.diag(np.array([1.0,] * self.num_collocation_points))

      p = np.zeros(self.decision_variable_size)

      self.qp = QuadraticProgram(Q, p)

   def linearized_dynamics_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
      '''
      Returns the `A` matrix and `b` vector of dynamics equality constraints.
      If the object's relinearization sequence is `None`, then the system's
      dynamics are linearized about the initial state and average of the system's
      minimum and maximum control inputs.
      '''
      relin_seq = None
      if self.relinearization_sequence is None:
         relin_seq = [
            (self.initial_state, (self.dynamics._u_max + self.dynamics._u_min) / 2.0),
         ] * self.num_collocation_points
      else:
         relin_seq = self.relinearization_sequence

      dt = self.time_horizon / self.num_collocation_points
      linear_terms: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
      for i, (state, control_input) in enumerate(relin_seq):
         time = dt * i
         linear_terms.append(self.dynamics.x_dot_linear(state, control_input, time))

      num_entries = self.num_collocation_points * self.state_size if self.final_state is None else (self.num_collocation_points + 1) * self.state_size

      A_eq = np.zeros(
         (
            self.num_collocation_points * self.state_size,
            self.num_collocation_points * (self.state_size + self.control_size)
         )
      )

      b_eq = np.zeros(self.num_collocation_points * self.state_size)

      control_input_offset = self.control_size * self.num_collocation_points

      # x_{k+1} = phi * x_k + phi_integ * B * u_k + phi * p

      for i, linear_tuple in enumerate(linear_terms):
         A_dyn, B_dyn, p_dyn = linear_tuple
         phi, phi_integ = phi_and_phi_integ(A_dyn, dt)

         # control input k
         A_eq[
            i * self.state_size : i * self.state_size + self.state_size,
            i * self.control_size : i * self.control_size + self.control_size
         ] = -np.dot(phi_integ, B_dyn)

         # state k
         A_eq[
            i * self.state_size : i * self.state_size + self.state_size,
            control_input_offset + i * self.state_size : control_input_offset + i * self.state_size + self.state_size
         ] = -phi

         # state k + 1
         A_eq[
            i * self.state_size : i * self.state_size + self.state_size,
            control_input_offset + (i + 1) * self.state_size : control_input_offset + (i + 1) * self.state_size + self.state_size
         ] = np.eye(self.state_size)

         b_eq[i * self.state_size : i * self.state_size + self.state_size] = np.dot(phi_integ, p_dyn)
