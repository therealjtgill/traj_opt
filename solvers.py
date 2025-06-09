import numpy as np
from quadratic_program import QuadraticProgram

class InfeasibleStartNewton:
   '''
   Implementation of Infeasible Start Newton's Method for a convex optimization
   problem with linear equality constraints.
   '''
   def __init__(self, qp: QuadraticProgram):
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
      alpha: float,
      beta: float,
      max_num_iters: int,
      eps: float=1e-8
   ):
      '''
      x_init - Initial value for the decision variable, `x`. The initial guess
         must be in the domain of the problem definition
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

      x = np.copy(x_init)
      v = np.copy(v_init)

      t = 20.0
      residual = self.residual(x, v, t)

      num_iters = 0

      while np.linalg.norm(residual) >= eps and (num_iters < max_num_iters):
         residual = self.residual(x, v, t)
         K = self.qp.kkt_matrix(x, t)

         delta_x_v = np.linalg.solve(K, -residual)
         delta_x = delta_x_v[0:self.qp.N]
         delta_v = delta_x_v[self.qp.N:]

         s = 1.0
         while (
            not self.qp.in_domain(x + s * delta_x)
            or np.linalg.norm(self.residual(x + s * delta_x, v + s * delta_v, t)) > (1.0 - alpha * s) * np.linalg.norm(residual)
         ):
            s *= beta

         x += s * delta_x
         v += s * delta_v

         num_iters += 1

      return x, v, np.linalg.norm(self.residual(x, v, t))
