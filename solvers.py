import copy
import numpy as np
import scipy
from quadratic_program import QuadraticProgram, BoxInequalityQuadraticProgram

class InfeasibleStartNewton:
   '''
   Implementation of Infeasible Start Newton's Method for a convex optimization
   problem with linear equality constraints. 
   '''
   def __init__(self, qp: QuadraticProgram | BoxInequalityQuadraticProgram):
      self.qp = qp

   def residual(self, x: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
      '''
      The `residual` that results from the KKT conditions applied to a convex
      optimization problem with linear equality constraints. In this case, it's
      the vector:\n
      \| grad(f(x)) + A^T v |\n
      \| Ax - b             |\n
      where `v` is the Lagrange multiplier vector for equality constraints.
      '''
      return np.hstack(
         [
            self.qp.gradient(x, t) + np.dot(self.qp.A.transpose(), v),
            np.dot(self.qp.A, x) - self.qp.b
         ]
      )

   def solve(
      self,
      x_init: np.ndarray,
      v_init: np.ndarray,
      t: float,
      alpha: float = 0.5,
      beta: float = 0.9,
      max_num_iters: int = 100,
      eps: float=1e-8
   ):
      '''
      x_init - Initial value for the decision variable, `x`. The initial guess
         must be in the domain of the problem definition, but it does not have
         to be feasible.
      v_init - Initial value for the equality constraint Lagrange multiplier, `v`
      alpha - Parameter for line search over residual values
      beta - Parameter to reduce line search magnitude by
      max_num_iters - Maximum number of iterations to run the solver while the
         residual is greater than some tolerance epsilon
      eps - Tolerance epsilon for early termination; the solver terminates if the
         L2 norm of the residual falls below this value
      '''
      assert(len(v_init.shape) == 1)
      assert(v_init.shape[0] == self.qp.M)
      assert(alpha > 0)
      assert(beta > 0)
      assert(max_num_iters > 0)

      x = copy.deepcopy(x_init)
      v = copy.deepcopy(v_init)

      res_norm_prev = np.inf
      res = self.residual(x, v, t)
      res_norm = np.linalg.norm(res)
      num_iters = 0

      print("stupid infeasible newton termination criteria:", (abs(res_norm - res_norm_prev) / (res_norm if np.isinf(res_norm_prev) else max(res_norm, res_norm_prev))))

      while (
         res_norm >= eps
         and (num_iters < max_num_iters)
         and (abs(res_norm - res_norm_prev) / (res_norm if np.isinf(res_norm_prev) else max(res_norm, res_norm_prev))) > 1e-6
      ):
         res_norm_prev = copy.deepcopy(res_norm)
         K = self.qp.kkt_matrix(x, t)

         delta_x_v = scipy.linalg.solve(K, -res, assume_a='sym')
         delta_x = delta_x_v[0:self.qp.N]
         delta_v = delta_x_v[self.qp.N:]

         s = 1.0
         while (
            not self.qp.in_domain(x + s * delta_x)
            or np.linalg.norm(self.residual(x + s * delta_x, v + s * delta_v, t)) > (1.0 - alpha * s) * np.linalg.norm(res)
         ):
            s *= beta

         x += s * delta_x
         v += s * delta_v
         res = self.residual(x, v, t)
         res_norm = np.linalg.norm(res)
         print("ifsnm num iters:", num_iters, "residual norm:", np.linalg.norm(res), "line search param:", s)

         num_iters += 1

      return x, v, np.linalg.norm(self.residual(x, v, t))

class FeasibleStartNewton:
   def __init__(self, qp: QuadraticProgram | BoxInequalityQuadraticProgram):
      self.qp = qp

   def newton_decrement(self, x: np.ndarray, t: float) -> float:
      pass

   def solve(
      self,
      x_init: np.ndarray,
      v_init: np.ndarray,
      t: float,
      alpha: float = 0.5,
      beta: float = 0.9,
      max_num_iters: int = 100,
      eps: float=1e-8
   ):
      assert(len(v_init.shape) == 1)
      assert(v_init.shape[0] == self.qp.M)
      assert(alpha > 0)
      assert(beta > 0)
      assert(max_num_iters > 0)

      x = copy.deepcopy(x_init)
      v = copy.deepcopy(v_init)

      num_iters = 0

      grad = self.qp.gradient(x, t)
      hess = self.qp.hessian(x, t)
      K = self.qp.kkt_matrix(x, t)
      delta_x_v = scipy.linalg.solve(K, np.hstack([-grad, np.zeros((self.qp.M,))]), assume_a='sym')
      delta_x = delta_x_v[0:self.qp.N]
      v = delta_x_v[self.qp.N:]

      newton_dec_prev = np.inf
      newton_dec = np.sqrt(np.dot(np.dot(delta_x, hess), delta_x))

      print("entering fsnm loop")
      print("newton decrement:", newton_dec)

      while (
         newton_dec >= (2.0 * eps)
         and (num_iters < max_num_iters)
         and (abs(newton_dec_prev - newton_dec) / (newton_dec if np.isinf(newton_dec_prev) else max(newton_dec_prev, newton_dec))) > 1e-6
      ):
         obj = self.qp.objective(x, t)
         grad = self.qp.gradient(x, t)
         hess = self.qp.hessian(x, t)
         K = self.qp.kkt_matrix(x, t)
         delta_x_v = scipy.linalg.solve(K, np.hstack([-grad, np.zeros((self.qp.M,))]), assume_a='sym')
         delta_x = delta_x_v[0:self.qp.N]
         v = delta_x_v[self.qp.N:]

         newton_dec_prev = copy.deepcopy(newton_dec)

         s = 1.0
         while (
            not self.qp.in_domain(x + s * delta_x)
            or self.qp.objective(x + s * delta_x, t) > obj + alpha * s * np.dot(grad, delta_x)
         ):
            s *= beta

         x = x + s * delta_x

         print("fsnm num iters:", num_iters, "line search:", s, "newton decrement:", newton_dec)
         grad = self.qp.gradient(x, t)
         hess = self.qp.hessian(x, t)
         newton_dec = np.sqrt(np.dot(np.dot(delta_x, hess), delta_x))

         # print("nd prev:", newton_dec_prev, "nd:", newton_dec)

         num_iters += 1

      return x, v, (num_iters > 0)
