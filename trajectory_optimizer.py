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
      effort_weights: np.ndarray,
      num_collocation_points: int,
      time_horizon: float,
      initial_state: np.ndarray,
      dynamics: CartpoleDynamics,
      final_state: np.ndarray | None = None,
      state_weights: np.ndarray | None = None,
      reference_trajectory: List[Tuple[np.ndarray, np.ndarray]] | None = None, # (state, input)
      relinearization_sequence: List[np.ndarray] | None = None
   ):
      assert(num_collocation_points > 0)
      assert(time_horizon > 0.0)
      assert(reference_trajectory is None or len(reference_trajectory) == num_collocation_points)
      assert(relinearization_sequence is None or len(relinearization_sequence) == num_collocation_points)
      assert(len(effort_weights.shape) == 1)
      assert(effort_weights.shape[0] == num_collocation_points * dynamics.control_size)
      assert(final_state is None or (len(final_state.shape) == 1 and final_state.shape[0] == dynamics.state_size))
      assert(
         len(effort_weights.shape) == 1
         and effort_weights.shape[0] == num_collocation_points * dynamics.control_size
      )
      assert(
         state_weights is None
         or (
            len(state_weights.shape) == 1
            and state_weights.shape[0] == num_collocation_points * dynamics.state_size
         )
      )

      self.dynamics: CartpoleDynamics = dynamics
      self.num_collocation_points: int = num_collocation_points
      self.time_horizon: float = time_horizon
      self.initial_state: np.ndarray = initial_state
      self.final_state: np.ndarray | None = final_state
      self.reference_trajectory: List[Tuple[np.ndarray, np.ndarray]] | None = reference_trajectory
      self.relinearization_sequence: List[np.ndarray] | None = relinearization_sequence

      self.state_size: int = self.dynamics.state_size
      self.control_size: int = self.dynamics.control_size

      # Number of elements in the decision variable
      self.decision_variable_size = self.num_collocation_points * (self.state_size + self.control_size)
      # Number of elements in the decision variable related to control input
      self.decision_variable_control_size = self.num_collocation_points * self.control_size
      # Number of elements in the decision variable related to dynamic state
      self.decision_variable_state_size = self.num_collocation_points * self.state_size

      # decision variable layout is
      # [control inputs, states]

      Q = np.zeros((self.decision_variable_size, self.decision_variable_size))
      Q[
         0: self.decision_variable_control_size,
         0: self.decision_variable_control_size
      ] = np.diag(effort_weights)

      p = np.zeros(self.decision_variable_size)

      A_eq, b_eq = self.equality_constraints()
      C_ineq, d_ineq = self.inequality_constraints()
      self.qp = QuadraticProgram(Q, p, A_eq, b_eq, C_ineq, d_ineq)

      self.isnm = InfeasibleStartNewton(self.qp)
      self.fsnm = FeasibleStartNewton(self.qp)

   def equality_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
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
         ] * (self.num_collocation_points)
      else:
         relin_seq = self.relinearization_sequence

      dt = self.time_horizon / self.num_collocation_points
      linear_terms: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
      for i, relin_vals in enumerate(relin_seq):
         state, control_input = relin_vals
         time = dt * i
         linear_terms.append(self.dynamics.x_dot_linear(state, control_input, time))

      num_entries = 0
      if self.final_state is not None:
         num_entries = (self.num_collocation_points + 1) * self.state_size
      else:
         num_entries = self.num_collocation_points * self.state_size

      A_eq = np.zeros((num_entries, self.decision_variable_size))
      b_eq = np.zeros(num_entries)

      control_input_offset = self.decision_variable_control_size

      # x_{i+1} = phi * x_i + phi_integ * B * u_i + phi * p

      for i, linear_tuple in enumerate(linear_terms):
         A_dyn, B_dyn, p_dyn = linear_tuple
         phi, phi_integ = phi_and_phi_integ(A_dyn, dt)

         # index directly into the A matrix at the control input location if the
         # control input's size is one, otherwise slice into the A matrix

         # control input i
         if self.control_size == 1:
            A_eq[
               i * self.state_size : i * self.state_size + self.state_size,
               i * self.control_size
            ] = -np.dot(phi_integ, B_dyn)
         else:
            A_eq[
               i * self.state_size : i * self.state_size + self.state_size,
               i * self.control_size : i * self.control_size + self.control_size
            ] = -np.dot(phi_integ, B_dyn)

         # the state at i = 1 is not part of the decision variable, so it needs
         # to be handled separately

         if i == 0:
            # state i
            A_eq[
               i * self.state_size : i * self.state_size + self.state_size,
               control_input_offset + i * self.state_size : control_input_offset + (i + 1) * self.state_size
            ] = np.eye(self.state_size)

            # affine portion of linearization plus dot(phi, x_init)
            b_eq[i * self.state_size : i * self.state_size + self.state_size] = np.dot(phi_integ, p_dyn) + np.dot(phi, self.initial_state)
         else:
            # state i - 1
            A_eq[
               i * self.state_size : i * self.state_size + self.state_size,
               control_input_offset + (i - 1) * self.state_size : control_input_offset + i * self.state_size
            ] = -phi

            # state i
            A_eq[
               i * self.state_size : i * self.state_size + self.state_size,
               control_input_offset + i * self.state_size : control_input_offset + (i + 1) * self.state_size
            ] = np.eye(self.state_size)

            b_eq[i * self.state_size : i * self.state_size + self.state_size] = np.dot(phi_integ, p_dyn)

      if self.final_state is not None:
         A_eq[
            self.num_collocation_points * self.state_size: self.num_collocation_points * self.state_size + self.state_size,
            control_input_offset + (self.num_collocation_points - 1) * self.state_size: control_input_offset + self.num_collocation_points * self.state_size,
         ] = np.eye(self.state_size)

         b_eq[self.num_collocation_points * self.state_size: self.num_collocation_points * self.state_size + self.state_size] = self.final_state

      return A_eq, b_eq

   def inequality_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
      C_ineq = np.zeros(
         (
            2 * self.decision_variable_control_size,
            self.decision_variable_size
         )
      )

      d_ineq = np.zeros(2 * self.decision_variable_control_size,)

      # u <= u_max
      C_ineq[
         0:self.decision_variable_control_size,
         0:self.decision_variable_control_size
      ] = np.eye(self.decision_variable_control_size)

      d_ineq[0:self.decision_variable_control_size] = self.dynamics._u_max

      # u >= u_min
      C_ineq[
         self.decision_variable_control_size: 2 * self.decision_variable_control_size,
         0:self.decision_variable_control_size
      ] = -np.eye(self.decision_variable_control_size)

      d_ineq[self.decision_variable_control_size: 2 * self.decision_variable_control_size] = -self.dynamics._u_min

      return C_ineq, d_ineq

   def gogogo(
      self,
      x_init: np.ndarray,
      v_init: np.ndarray,
      barrier_t: float = 20.0,
      barrier_lambda: float = 10.0,
      residual_threshold: float = 1e-5,
      max_num_ipm_iters: int = 10,
      max_num_newton_iters: int = 100
   ) -> Tuple[np.ndarray, np.ndarray, bool]:
      assert(len(x_init.shape) == 1)
      assert(len(v_init.shape) == 1)

      assert(x_init.shape[0] == self.decision_variable_size)
      if self.final_state is None:
         assert(v_init.shape[0] == self.decision_variable_state_size)
      else:
         assert(v_init.shape[0] == self.decision_variable_state_size + self.state_size)

      x_out, v_out, residual = self.isnm.solve(
         x_init, v_init, barrier_t, 0.5, 0.9, max_num_newton_iters
      )

      if residual > residual_threshold:
         print("Infeasible with ISNM residual", residual)
         return x_out, v_out, False
      else:
         print("Feasible with ISNM residual", residual)

      for _ in range(max_num_ipm_iters):
         barrier_t *= barrier_lambda
         x_out, v_out, keep_going = self.fsnm.solve(x_out, v_out, barrier_t, 0.5, 0.9, max_num_newton_iters)
         if not keep_going:
            break

      return x_out, v_out, True

   def get_states(self, x: np.ndarray) -> List[np.ndarray]:
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.decision_variable_size)
      states = np.reshape(x[self.decision_variable_control_size:], (-1, self.state_size))

      states_list = [z for z in states]

      return states_list

   def get_inputs(self, x: np.ndarray) -> List[np.ndarray]:
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.decision_variable_size)
      inputs = np.reshape(x[:self.decision_variable_control_size], (-1, self.control_size))
      inputs_list = [u for u in inputs]

      return inputs_list
