import copy
import numpy as np

def rk4(x_init: np.ndarray, t_init: float, dt: float, f):
   '''
   Integrates the dynamics of `x_dot = f(x, t)` over a time interval `dt` given an initial state, `x_init`, and time, `t_init`
   using the RK4 algorithm.
   Returns the state at the end of the time interval `dt`.

   - x_init = Initial system state
   - t_init = Initial time
   - dt = Time interval over which `f` is integrated
   - f = f(t, x) - the function representing the time derivative of the state vector, `x`
   '''
   k1 = f(t_init, x_init)
   k2 = f(t_init + dt/2, x_init + dt * k1/2)
   k3 = f(t_init + dt/2, x_init + dt * k2/2)
   k4 = f(t_init + dt, x_init + dt * k3)

   return x_init + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def phi_and_phi_integ(A: np.ndarray, dt: float):
   '''
   Returns the state transition matrix and integral of the state transition matrix
   for an LTI system. This is meant to be used to calculate the discrete state transition
   equations for systems with constant input over the interval `dt`.

   - A = Square matrix describing a system's linearized dynamics
   - dt = Small time period
   '''
   def mat_exp_ish(A_, dt_):
      '''
      Approximates the matrix exponential of e^(A dt) by expanding the infinite summation
      over a finite number of terms. The number of terms in the expansion is given by:

      -15/log(dt)
      '''
      prev_mult = np.eye(np.shape(A_)[0], np.shape(A_)[1])
      sum = prev_mult
      for n in range(int(np.log(1e-15)/np.log(dt_))):
         mult = np.dot(prev_mult, A_) * dt_ / (n + 1)
         sum += mult
         prev_mult = mult
      return sum

   def mat_exp_ish_integral(A_, dt_):
      '''
      Approximates the integratl of the matrix exponential of e^(A dt) by expanding the infinite summation
      over a finite number of terms. The number of terms in the expansion is given by:

      -15/log(dt)
      '''
      prev_mult = np.eye(np.shape(A_)[0], np.shape(A_)[1]) * dt_
      sum = prev_mult
      for n in range(int(np.log(1e-15)/np.log(dt_))):
         mult = np.dot(prev_mult, A_) * dt_ / (n + 2)
         sum += mult
         prev_mult = mult
      return sum

   return mat_exp_ish(A, dt), mat_exp_ish_integral(A, dt)

def lti_sim_with_input(F: np.ndarray, b: np.ndarray, f: np.ndarray, u_func, z_init, dt, t_0, t_f):
    '''
    Generates a dense set of states given some linearized dynamics, control input, and initial
    state.

    - F = Square linearized dynamics matrix, z_dot = F * z + b * u
    - b = Vector that applies a scalar control input to some system state in the linearized system dynamics
    - f = Affine portion of linearization of system dynamics about a non-fixed point
    - u_func = A function that returns a control input given a time value
    - z_init = Initial cartpole system state
    - dt = Time duration over which the LTI system is simulated
    - t_0 = Start time
    - t_f = End time
    - state_plotter_func
    '''
    zs = [copy.deepcopy(z_init)]

    t = t_0
    phi, phi_integ = phi_and_phi_integ(F, dt)

    while t < t_f:
        u_now = u_func(t)
        zs.append(np.dot(phi, zs[-1]) + np.dot(phi_integ, b * u_now + f))
        t += dt

    print("final state:", zs[-1])

    return zs
