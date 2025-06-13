import numpy as np
from typing import List, Tuple

from dynamics import rk4

class CartpoleDynamics:
   def __init__(
      self,
      pendulum_length: float,
      pendulum_mass: float,
      cart_mass: float,
      control_input_min: float,
      control_input_max: float
   ):
      self._l_pend = pendulum_length
      self._m_pend = pendulum_mass
      self._m_cart = cart_mass
      self._u_min = control_input_min
      self._u_max = control_input_max

   def state_size(self) -> int:
      return 4

   def control_size(self) -> int:
      return 1

   def state_description(self) -> List[str]:
      return ["cart position", "pendulum angle", "cart position dot", "pendulum angle dot"]

   def x_dot_nonlinear(self, state: np.ndarray, control_input: float, time: float):
      '''
      Returns the time derivative of the state of a cartpole system given some current time, state, and control input.

      - time = Time (ignored)
      - state = Cartpole system state, [cart_pos, pend_angle, cart_pos_speed, pend_angle_speed]
      - control_input = Force exerted on cart, clamped between +/-u_max

      For x_dot = f(x, u, t), returns f(state, control_input, time).
      '''
      assert(len(state.shape) == 1)
      assert(state.shape[0] == 4)

      u = np.clip(control_input, self._u_min, self._u_max)
      cosz1 = np.cos(state[1])
      sinz1 = np.sin(state[1])
      alpha = (self._m_pend + self._m_cart) / (self._m_pend * self._l_pend)
      g = 9.8
      denom = (cosz1 * cosz1 - self._l_pend * alpha)

      return np.array(
         [
            state[2],
            state[3],
            (-g * sinz1 * cosz1 - self._l_pend * u - self._l_pend * (state[3]**2) * sinz1) / denom,
            (alpha * g * sinz1 + u * cosz1 + (state[3]**2 * sinz1 * cosz1)) / denom
         ]
      )

   def step(
      self,
      state: np.ndarray,
      control_input: float,
      dt: float,
      time: float
   ) -> np.ndarray:
      assert(len(state.shape) == 1)
      assert(state.shape[0] == 4)

      zoh_input_dynamics = lambda t, y : self.x_dot_nonlinear(y, control_input, t)

      new_state = rk4(state, time, dt, zoh_input_dynamics)

      return new_state

   def x_dot_linear(
      self,
      state: np.ndarray,
      control_input: float,
      time: float
   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
      '''
      linearizes the cartpole dynamics about `z` and `u`, returns the `A` and `b` matrices
         `z_dot = f(t, z, u) ~ f(t_0, z_0, u_0) + grad(f, z)(t_0, z_0, u_0) ^ T (z - z_0) + grad(f, u)(t_0, z_0, u_0) ^ T (u - u_0)
         `A` - Jacobian of cartpole dynamics with respect to state, `z`
         `r` - affine part of the linear approximation of cartpole dynamics
         `b` - Jacobian of cartpole dynamics with respect to control input, `u`

      - state = [cart_pos, pend_angle, cart_pos_speed, pend_angle_speed]
      - control_input = force exerted on cart, clamped between +/-u_max
      
      Returns `A`, `b`, `r`
      '''
      assert(len(state.shape) == 1)
      assert(state.shape[0] == 4)

      A = np.zeros((4, 4))
      b = np.zeros(4)

      u = np.clip(control_input, self._u_min, self._u_max)
      cosz1 = np.cos(state[1])
      sinz1 = np.sin(state[1])
      alpha = (self._m_pend + self._m_cart) / (self._m_pend * self._l_pend)
      g = 9.8
      denom = (cosz1 * cosz1 - self._l_pend * alpha)

      A[0, 2] = 1.0
      A[1, 3] = 1.0
      A[2, 1] = ((g * sinz1 * sinz1 - g * cosz1 * cosz1 - (state[3] ** 2) * cosz1) / denom) + 2.0 * (((-g * sinz1 * cosz1 - self._l_pend * u - self._l_pend * (state[3] ** 2) * sinz1) * sinz1 * cosz1)) / (denom * denom)
      A[2, 3] = -2.0 * self._l_pend * state[3] * sinz1 / denom
      A[3, 1] = ((alpha * g * cosz1 - u * sinz1 - (state[3] ** 2) * (sinz1 * sinz1 - cosz1 * cosz1)) / denom) + 2.0 * (alpha * g * sinz1 + u * cosz1 + (state[3] ** 2) * sinz1 * cosz1) * (sinz1 * cosz1) / (denom * denom)
      A[3, 3] = 2.0 * state[3] * sinz1 * cosz1 / denom

      b[2] = -1.0 / denom
      b[3] = cosz1 / denom

      r = self.x_dot_nonlinear(state, u, time) - np.dot(A, state) - b * u

      return A, b, r
